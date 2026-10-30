"""Synthetic only: no actual archive/data/checkpoint paths are opened."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import entry as e

def fixture():
    cohort={str(s):dict(status='complete',**{k+suffix:('/synthetic/'+k if suffix=='_path' else 'a'*64)
            for k in ('recipe','teacher','replacement','validation','export','metrics') for suffix in ('_path','_sha256')}) for s in e.SEEDS}
    source={n:'b'*64 for n in e.SOURCE_FILES}
    return dict(status='parent-approved-and-user-announced',action='separately-reviewed-official-test-release',
        planned_seeds=list(e.SEEDS),updates=1000,threads=4,interop_threads=1,cap_seconds=7200,absolute_deadline=e.DEADLINE,
        claim_scope=e.CLAIM_SCOPE,no_adaptation=True,one_run=True,cohort=cohort,cohort_sha256=e.canonical(cohort),
        source_sha256=source,replacement_source_sha256={n:source[n] for n in e.prior.SOURCE_FILES},
        data_identity_sha256='c'*64,data_manifest_sha256='d'*64)

class Guards(unittest.TestCase):
    def test_valid_pure_guard_without_any_file_open(self):
        with patch('builtins.open',side_effect=AssertionError('No open')),patch.object(Path,'open',side_effect=AssertionError('No Path.open')):
            self.assertEqual(e.release_guard(fixture()),list(e.SEEDS))

    def test_denied_release_zero_file_opens(self):
        cases=[]
        for k,v in [('status','prepared'),('no_adaptation',False),('updates',999),('one_run',False),('threads',8),('cohort_sha256','0'*64),('source_sha256',{})]:
            p=fixture();p[k]=v;cases.append(p)
        for status in ('pending','running','fit-failed','teacher-incomplete','teacher-quality-failed'):
            p=fixture();p['cohort'][str(e.SEEDS[0])]['status']=status;p['cohort_sha256']=e.canonical(p['cohort']);cases.append(p)
        p=fixture();del p['cohort'][str(e.SEEDS[-1])];p['cohort_sha256']=e.canonical(p['cohort']);cases.append(p)
        with patch('builtins.open',side_effect=AssertionError('No open')),patch.object(Path,'open',side_effect=AssertionError('No Path.open')):
            for p in cases:
                with self.subTest(p=p),self.assertRaises(ValueError): e.release_guard(p)

    def test_source_mismatch_precedes_all_artifact_and_data_loading(self):
        p=fixture();slot=dict(status='parent-reserved-free-slot',plan_sha256='e'*64,threads=4,other_science_threads=4,global_science_threads=8,expires_at=1e20)
        with patch.object(e,'verified_bytes',side_effect=ValueError('source mismatch')) as read,patch.object(e,'numerical_modules',side_effect=AssertionError('No numerical imports')):
            with self.assertRaises(ValueError):e.verify(p,'e'*64,slot)
            self.assertEqual(read.call_count,1)
            self.assertTrue(str(read.call_args.args[0]).startswith(str(e.ROOT)))

    def test_hash_rejection_synthetic_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'fake';p.write_bytes(b'not a checkpoint')
            with self.assertRaises(ValueError):e.verified_bytes(p,'0'*64)

    def test_registry_file_and_parent_fsync_and_no_second_output(self):
        import os,stat
        with tempfile.TemporaryDirectory() as tmp:
            registry=Path(tmp)/'one-look.json'; events=[]; original=os.fsync
            def sync(fd):
                events.append('directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file')
                return original(fd)
            with patch.object(e,'REGISTRY',registry),patch.object(e.os,'fsync',side_effect=sync):
                e.claim_one_run(Path(tmp)/'first','a'*64)
                self.assertEqual(events,['file','directory'])
                for output in ('first','different-output'):
                    with self.assertRaises(FileExistsError):e.claim_one_run(Path(tmp)/output,'b'*64)
                registry.write_bytes(b'') # synthetic crash/partial record, never production reset
                with self.assertRaises(FileExistsError):e.claim_one_run(Path(tmp)/'third','c'*64)

    def test_registry_directory_fsync_failure_blocks_bootstrap(self):
        import os,stat
        with tempfile.TemporaryDirectory() as tmp:
            p=fixture();p['output']=str(Path(tmp)/'output'); registry=Path(tmp)/'one-look.json'
            original=os.fsync
            def sync(fd):
                if stat.S_ISDIR(os.fstat(fd).st_mode): raise OSError('synthetic directory fsync failure')
                return original(fd)
            with patch.object(e,'REGISTRY',registry),patch.object(e.prior,'checked_json',side_effect=[p,{}]),patch.object(e,'verify',return_value=list(e.SEEDS)),patch.object(e.os,'fsync',side_effect=sync),patch.object(e,'bootstrap',side_effect=AssertionError('must not bootstrap')) as boot:
                with self.assertRaises(OSError):e.execute('fake-plan','a'*64,'fake-slot','b'*64)
                boot.assert_not_called();self.assertTrue(registry.exists())

class Numerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from . import evaluate
        cls.v=evaluate;cls.t=torch

    def test_admission_failure_never_opens_test(self):
        p=fixture()
        with patch.object(self.v,'admit_all',side_effect=ValueError('not admitted')),patch.object(self.v,'load_test',side_effect=AssertionError('TEST MUST STAY SEALED')) as loader:
            with self.assertRaises(ValueError):self.v.run(p,list(e.SEEDS),{},Path('/synthetic'),{'expiry':1e20})
            loader.assert_not_called()

    def test_full_n_centering_and_exact_paired_counts(self):
        import numpy as np
        y=self.t.tensor([1.,2.,5.,8.]).reshape(4,1,1,1)
        stats={n:self.v.c.EndpointStats() for n in self.v.ARMS}
        for s in stats.values():
            s.update(y[:1]+1,y[:1]);s.update(y[1:]+1,y[1:])
            self.assertAlmostEqual(s.result()['centered_span_nmse'],4/(float(((y-y.mean())**2).sum())+4e-12))
        labels=np.array([0,1,0,1]);pred={'teacher':labels.copy(),'Comp-joint':np.array([0,1,0,0]),'Sep-local':np.array([1,1,0,0]),'Sep-joint':labels.copy()}
        out=self.v.summarize(pred,labels,stats)
        self.assertEqual(out['teacher_correct'],4);self.assertEqual(out['arms']['Comp-joint']['correct'],3)
        self.assertEqual(out['arms']['Comp-joint']['teacher_only_correct'],1)
        self.assertEqual(out['arms']['Comp-joint']['paired_degradation_pp'],25)
        self.assertEqual(len(out['contrasts']),3)
        with self.assertRaises(ValueError):stats['Comp-joint'].update(y*self.t.nan,y)

    def fit_fixture(self):
        seed=e.SEEDS[0];ids=tuple(range(5000)); arms=self.v.c.make_arms(seed)
        state=dict(cursor=1000,initial={n:copy.deepcopy(a.state_dict()) for n,a in arms.items()},optimizers={})
        for name,arm in arms.items():
            opt=self.v.c.make_optimizer(arm).state_dict()
            pids=[i for group in opt['param_groups'] for i in group['params']]
            opt['state']={i:dict(step=self.t.tensor(1000.),exp_avg=self.t.zeros_like(p),exp_avg_sq=self.t.zeros_like(p)) for i,p in zip(pids,arm.parameters())}
            state['optimizers'][name]=opt
        lock=dict(initialization={n:self.v.r.tensor_hash(a.state_dict()) for n,a in arms.items()},
                  batch_schedule_sha256=self.v.c.canonical_hash(self.v.c.calibration_batches(ids,seed)))
        return state,lock,ids,seed

    def test_reconstructed_frozen_fit_and_rejections_before_test(self):
        state,lock,ids,seed=self.fit_fixture()
        self.assertEqual(set(self.v.verify_frozen_fit(state,lock,ids,seed)),set(self.v.ARMS))
        cases=[]
        a,b=copy.deepcopy(state),copy.deepcopy(lock);b['batch_schedule_sha256']='0'*64;cases.append((a,b))
        a,b=copy.deepcopy(state),copy.deepcopy(lock);next(iter(a['initial']['Comp-joint'].values())).add_(1);b['initialization']['Comp-joint']=self.v.r.tensor_hash(a['initial']['Comp-joint']);cases.append((a,b))
        a,b=copy.deepcopy(state),copy.deepcopy(lock);a['optimizers']['Comp-joint']['param_groups'][0]['lr']=.02;cases.append((a,b))
        a,b=copy.deepcopy(state),copy.deepcopy(lock);a['optimizers']['Comp-joint']['state'][0]['step']=self.t.tensor(999.);cases.append((a,b))
        a,b=copy.deepcopy(state),copy.deepcopy(lock);del a['optimizers']['Comp-joint']['state'][0];cases.append((a,b))
        for a,b in cases:
            def denied(*args):return self.v.verify_frozen_fit(a,b,ids,seed)
            with patch.object(self.v,'admit_all',side_effect=denied),patch.object(self.v,'load_test',side_effect=AssertionError('test must remain sealed')) as loader:
                with self.assertRaises(ValueError):self.v.run(fixture(),list(e.SEEDS),{},Path('/synthetic'),{'expiry':1e20})
                loader.assert_not_called()

    def test_safe_unpickler_rejects_unapproved_global(self):
        import io,pickle
        with self.assertRaises(pickle.UnpicklingError):self.v.d.SafeUnpickler(io.BytesIO(b'cos\nsystem\n.')).load()

if __name__=='__main__':unittest.main()
