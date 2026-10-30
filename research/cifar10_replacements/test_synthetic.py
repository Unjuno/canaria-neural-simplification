"""Tiny synthetic-only checks: never import teacher runner or open real artifacts."""
import copy
import inspect
import io
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
from . import core as c

torch.set_num_threads(1)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)

    def test_architecture_parameters_and_initialization(self):
        arms = c.make_arms(10)
        for name, arm in arms.items():
            self.assertEqual(sum(p.numel() for p in arm.parameters()), 5120)
            self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in arm.modules()))
            x = torch.randn(2,32,16,16)
            self.assertEqual(arm(x).shape, x.shape)
            self.assertTrue((arm(x) >= 0).all())
            for m in arm:
                self.assertIsNone(m.down.bias); self.assertIsNone(m.up.bias)
                self.assertTrue(torch.equal(m(x), F.relu(x+m.up(F.relu(m.down(x))))))
        for a,b in zip(arms['Sep-local'].parameters(), arms['Sep-joint'].parameters()):
            self.assertTrue(torch.equal(a,b)); self.assertNotEqual(a.data_ptr(),b.data_ptr())
        opts = [c.make_optimizer(arms[n]) for n in ('Sep-local','Sep-joint')]
        self.assertEqual(opts[0].state_dict(),opts[1].state_dict())

    def test_teacher_eval_bn_and_installed_data_path(self):
        teacher = c.FrozenTeacher(c.ResNet20())
        teacher.train(True)
        before = copy.deepcopy(teacher.model.state_dict())
        x = torch.randn(2,3,32,32)
        endpoints = teacher.features(x)
        self.assertTrue(all(t.shape == (2,32,16,16) for t in endpoints))
        self.assertTrue(all(not t.requires_grad for t in endpoints))
        self.assertTrue(torch.equal(teacher.installed_logits(x, teacher.model.blocks[4:6]), teacher.model(x)))
        self.assertTrue(all(not m.training for m in teacher.modules()))
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        for key,value in before.items():
            self.assertTrue(torch.equal(value,teacher.model.state_dict()[key]))

    def test_local_teacher_forcing_and_gradient_scale(self):
        arms = c.make_arms(19)
        endpoints = tuple(torch.rand(2,32,4,4,requires_grad=True) for _ in range(3))
        local = arms['Sep-local']; joint = arms['Sep-joint']
        seen = []
        hook = local[1].register_forward_pre_hook(lambda m,args: seen.append(args[0].clone()))
        loss = c.objective('Sep-local',local,endpoints); loss.backward(); hook.remove()
        self.assertTrue(torch.equal(seen[0],endpoints[1]))
        # Sum, not mean, of local objectives preserves each map's standalone gradient.
        for i in range(2):
            standalone = copy.deepcopy(local[i]); standalone.zero_grad()
            F.mse_loss(standalone(endpoints[i].detach()),endpoints[i+1].detach()).backward()
            for a,b in zip(local[i].parameters(),standalone.parameters()):
                self.assertTrue(torch.equal(a.grad,b.grad))
        seen=[]
        hook=joint[1].register_forward_pre_hook(lambda m,args:seen.append(args[0].detach().clone()))
        c.objective('Sep-joint',joint,endpoints).backward(); hook.remove()
        self.assertTrue(torch.equal(seen[0],joint[0](endpoints[0]).detach()))
        self.assertFalse(torch.equal(seen[0],endpoints[1]))
        self.assertTrue(all(t.grad is None for t in endpoints))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in joint.parameters()))
        before=[p.clone() for p in local.parameters()]; c.make_optimizer(local).step()
        self.assertTrue(all(not torch.equal(a,b) for a,b in zip(before,local.parameters())))

    def test_schedule_nmse_and_no_reader(self):
        batches = c.calibration_batches(tuple(range(5000)),123)
        self.assertEqual(batches,c.calibration_batches(tuple(range(5000)),123))
        self.assertEqual(len(batches),1000)
        self.assertEqual(len(batches[39]),8)
        self.assertEqual(sorted(sum((list(b) for b in batches[:40]),[])),list(range(5000)))
        self.assertEqual(c.uncentered_energy_nmse(5,10),.5)
        with self.assertRaises(ValueError): c.uncentered_energy_nmse(1,0)
        # Core has no filesystem or dataset helper path. In-memory admission is below.
        source=inspect.getsource(c)
        for forbidden in ('open(', 'read_bytes(', 'read_text(', 'load_data', 'tarfile', 'pickle', 'cifar10_teacher.run'):
            self.assertNotIn(forbidden,source)

    def artifact(self):
        identity=dict(schema=1,model_config={'synthetic':True},model_config_sha256=c.canonical_hash({'synthetic':True}))
        identity.update(split_seed=20261026, split_indices_encoding='little-endian int64 C-order raw bytes',
                        splits={k:dict(n=n,sha256='synthetic') for k,n in [('teachertrain',40000),('calibration',5000),('validation',5000)]},
                        normalization=dict(source='teachertrain only',sha256='synthetic'))
        state=dict(schema=2,seed=c.SEEDS[0],epoch=120,cursor=0,order=[],binding='synthetic',
                   evaluator_identity=identity,environment={'synthetic':True},spec={'epochs':120},model=c.ResNet20().state_dict())
        buf=io.BytesIO(); torch.save(state,buf); payload=buf.getvalue()
        blob=b'CANARIA1\n'+c.sha(payload).encode()+b'\n'+payload
        export=dict(identity,seed=c.SEEDS[0],epoch=120,data_manifest_sha256='synthetic',checkpoint_sha256=c.sha(blob),
                    official_test_decoded=False,calibration_used=False,environment=state['environment'],spec=state['spec'])
        metrics=dict(seed=c.SEEDS[0],epoch=120,data_manifest_sha256='synthetic',checkpoint_sha256=c.sha(blob),
                     status='complete',adequate=True,n=5000,correct=4250,accuracy=.85,test_batch_decoded=False,
                     calibration_used=False,environment=state['environment'])
        return export,metrics,blob,identity,'synthetic',c.SEEDS[0]

    def test_checkpoint_identity_and_admission(self):
        args=self.artifact()
        with patch('builtins.open',side_effect=AssertionError('No filesystem allowed')):
            self.assertIsInstance(c.admit_teacher(*args),c.FrozenTeacher)
        for position,key,value in ((1,'accuracy',.8499),(1,'adequate',False),(1,'status','pending'),
                                   (0,'epoch',119),(1,'correct',4249),(1,'n',4999),(0,'seed',77),
                                   (0,'checkpoint_sha256','bad'),(0,'data_manifest_sha256','bad'),
                                   (0,'official_test_decoded',True),(1,'calibration_used',True),
                                   (0,'model_config_sha256','bad')):
            changed=list(args); changed[position]=dict(changed[position],**{key:value})
            with self.subTest(key=key,value=value), self.assertRaises(ValueError): c.admit_teacher(*changed)
        changed=list(args); changed[2]=args[2]+b'x'
        with self.assertRaises(ValueError):c.admit_teacher(*changed)


if __name__ == '__main__':
    unittest.main()
