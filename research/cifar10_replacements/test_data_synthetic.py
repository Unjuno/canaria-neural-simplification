"""Tiny synthetic loader checks; no production data or checkpoints opened."""
import io
import pickle
import tarfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import data as d
from . import core

torch.set_num_threads(1)


class FakeArchive:
    def __init__(self):
        self.names = [f'cifar-10-batches-py/data_batch_{i}' for i in range(1,6)]
        self.members = [tarfile.TarInfo(n) for n in self.names+['cifar-10-batches-py/test_batch']]
        self.read = []
    def getmembers(self): return self.members
    def getmember(self,name): return next(m for m in self.members if m.name == name)
    def extractfile(self,member):
        if member.name.endswith('test_batch'): raise AssertionError('Test decode forbidden')
        self.read.append(member.name)
        return io.BytesIO(b'synthetic')


class DataPreparationTests(unittest.TestCase):
    def test_default_guard_before_any_read(self):
        with patch('pathlib.Path.read_bytes',side_effect=AssertionError('No reads')):
            with self.assertRaises(PermissionError): d.load_approved_data('/never-open')

    def test_archive_only_five_training_member_requests(self):
        archive=FakeArchive()
        manifest={'train_member_sha256':{n:core.sha(b'synthetic') for n in archive.names}}
        # Broadcast view is only one byte of backing storage, not a 30MB allocation.
        record={b'data':np.broadcast_to(np.array(0,dtype=np.uint8),(10000,3072)),b'labels':[0]*10000}
        with patch.object(d,'SafeUnpickler') as unpickler, patch.object(d.np,'concatenate',return_value='sentinel'):
            unpickler.return_value.load.return_value=record
            self.assertEqual(d.decode_train_members(archive,manifest),('sentinel','sentinel'))
        self.assertEqual(archive.read,archive.names)
        archive.members.append(archive.members[0])
        with self.assertRaises(ValueError):d.decode_train_members(archive,manifest)

    def test_restricted_pickle_and_unsafe_archive(self):
        with self.assertRaises(pickle.UnpicklingError):
            d.SafeUnpickler(io.BytesIO(b'cos\nsystem\n.')).load()
        archive=FakeArchive(); archive.members[0].type=tarfile.SYMTYPE
        with self.assertRaises(ValueError):d.decode_train_members(archive,{})

    def test_partition_exact_membership_and_identity(self):
        labels=np.repeat(np.arange(10),5000)
        splits=d.partition(labels)
        self.assertEqual(len(set(np.concatenate(list(splits.values())))),50000)
        self.assertEqual({k:len(v) for k,v in splits.items()},dict(teachertrain=40000,calibration=5000,validation=5000))
        means=np.array([.1,.2,.3]); std=np.array([.2,.3,.4])
        identity=d.evaluator_identity(splits,means,std)
        data=d.ApprovedData(np.empty((0,3,32,32),dtype=np.uint8),labels,splits,means,std,identity,'synthetic')
        data.require_calibration_ids(splits['calibration'])
        with self.assertRaises(ValueError):data.require_calibration_ids(splits['validation'])
        with self.assertRaises(ValueError):data.require_calibration_ids(splits['calibration'][::-1])
        with self.assertRaises(ValueError):data.image_batch([splits['teachertrain'][0]],split='calibration')
        with self.assertRaises(ValueError):data.image_batch([],split='test')
        changed=d.evaluator_identity(splits,means+1,std)
        self.assertNotEqual(identity['normalization']['sha256'],changed['normalization']['sha256'])

    def test_synthetic_production_identity_admission(self):
        splits=d.partition(np.repeat(np.arange(10),5000))
        mean=np.array([.1,.2,.3]); std=np.array([.2,.3,.4])
        identity=d.evaluator_identity(splits,mean,std)
        binding=d.APPROVED_MANIFEST['preflight_sha256']['data_manifest.json']
        data=d.ApprovedData(None,None,splits,mean,std,identity,binding)
        env=dict(python='3.13.5',torch='2.10.0+cpu',numpy='2.3.5',source_sha256=d.TEACHER_SOURCE_SHA256,
                 threads=4,interop=1,deterministic=True,
                 thread_env={k:'4' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        state=dict(schema=2,seed=core.SEEDS[0],epoch=120,cursor=0,order=[],binding=binding,
                   evaluator_identity=identity,environment=env,spec=d.APPROVED_SPEC,model=core.ResNet20().state_dict(),
                   train=splits['teachertrain'].tolist(),mean=torch.tensor(mean,dtype=torch.float32)[None,:,None,None],
                   std=torch.tensor(std,dtype=torch.float32)[None,:,None,None])
        buf=io.BytesIO();torch.save(state,buf);payload=buf.getvalue()
        blob=b'CANARIA1\n'+core.sha(payload).encode()+b'\n'+payload
        export=dict(identity,seed=core.SEEDS[0],epoch=120,data_manifest_sha256=binding,
                    checkpoint='final_epoch120.ckpt',checkpoint_sha256=core.sha(blob),
                    spec=d.APPROVED_SPEC,environment=env,official_test_decoded=False,calibration_used=False)
        metrics=dict(seed=core.SEEDS[0],epoch=120,data_manifest_sha256=binding,checkpoint_sha256=core.sha(blob),
                     environment=env,status='complete',adequate=True,n=5000,correct=4250,accuracy=.85,
                     test_batch_decoded=False,calibration_used=False)
        with patch('builtins.open',side_effect=AssertionError('No file reads')):
            self.assertIsInstance(d.admit_approved_teacher(export,metrics,blob,data,core.SEEDS[0]),core.FrozenTeacher)
            bundle=d.admit_teacher_bundle(export,metrics,blob,data,core.SEEDS[0])
            self.assertEqual(bundle.checkpoint_sha256,core.sha(blob))
            self.assertEqual(bundle.identity_sha256,core.canonical_hash(data.identity))
            self.assertEqual(bundle.model_state_sha256,d.model_state_hash(bundle.teacher.model))
        changed=dict(export,spec=dict(d.APPROVED_SPEC,epochs=119))
        with self.assertRaises(ValueError):d.admit_approved_teacher(changed,metrics,blob,data,core.SEEDS[0])

    def test_teacher_pins_reject_before_core(self):
        data=d.ApprovedData(None,None,{},None,None,{'model_config':d.MODEL_CONFIG},
                            d.APPROVED_MANIFEST['preflight_sha256']['data_manifest.json'])
        export={'spec':d.APPROVED_SPEC,'environment':{'source_sha256':'tampered'}}
        with patch.object(core,'admit_teacher',side_effect=AssertionError('Must reject before generic admission')):
            with self.assertRaises(ValueError):d.admit_approved_teacher(export,{},b'',data,core.SEEDS[0])
        self.assertEqual(d.APPROVED_SPEC['epochs'],120)
        self.assertEqual(d.APPROVED_SPEC['threads'],4)
        self.assertEqual(d.APPROVED_MANIFEST['source_commit'],'67bd076ec10574dc372b86f94e1d937c8c324189')


if __name__=='__main__':unittest.main()
