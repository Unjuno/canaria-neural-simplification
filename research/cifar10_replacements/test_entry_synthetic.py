"""Only synthetic JSON/bytes and mocked numerical services; no science calls."""
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch
from . import entry as e


class EntryTests(unittest.TestCase):
    def fixture(self,root,correct=4250):
        seed=e.SEEDS[0]; blob=b'synthetic checkpoint'; (root/'teacher.ckpt').write_bytes(blob)
        export=dict(seed=seed,epoch=120,checkpoint_sha256=e.digest(blob),official_test_decoded=False,calibration_used=False)
        metrics=dict(seed=seed,epoch=120,checkpoint_sha256=e.digest(blob),status='complete',adequate=correct>=4250,
                     n=5000,correct=correct,accuracy=correct/5000,test_batch_decoded=False,calibration_used=False)
        for name,record in [('export',export),('metrics',metrics)]: (root/(name+'.json')).write_text(json.dumps(record))
        run=dict(output=str(root/'output'),teacher_export=str(root/'export.json'),
                 teacher_export_sha256=e.digest((root/'export.json').read_bytes()),teacher_metrics=str(root/'metrics.json'),
                 teacher_metrics_sha256=e.digest((root/'metrics.json').read_bytes()),teacher_checkpoint=str(root/'teacher.ckpt'),
                 teacher_checkpoint_sha256=e.digest(blob))
        plan=dict(status='parent-approved-and-user-announced',planned_seeds=list(e.SEEDS),threads=4,interop_threads=1,
                  max_replacement_workers=1,global_science_threads=8,per_teacher_cap_seconds=e.CAP_SECONDS,
                  absolute_deadline=e.DEADLINE,official_test='sealed',updates=1000,preflight='/synthetic-only',
                  source_sha256={name:e.digest((e.ROOT/name).read_bytes()) for name in e.SOURCE_FILES},
                  runs={str(s):dict(run,output=str(root/('output' if s==seed else str(s)))) for s in e.SEEDS})
        pp=root/'plan.json'; pp.write_text(json.dumps(plan)); ph=e.digest(pp.read_bytes())
        slot=dict(status='parent-reserved-free-slot',plan_sha256=ph,seed=seed,output=run['output'],replacement_threads=4,
                  other_science_threads=4,global_science_threads=8,resume_authorized=False,expires_at=time.time()+60)
        sp=root/'slot.json';sp.write_text(json.dumps(slot))
        return (pp,ph,sp,e.digest(sp.read_bytes()),seed),run

    def test_exact_call_order_and_success_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,run=self.fixture(Path(tmp)); calls=[]
            worker=types.SimpleNamespace(close=lambda:calls.append('close'))
            def boot(*a,**kw):calls.append('bootstrap');return {'synthetic':True},worker
            def load(*a,**kw):calls.append('load-calibration');return 'data'
            def admit(*a,**kw):calls.append('admission');return 'bundle'
            def fit(data,bundle,output,*a,**kw):
                self.assertEqual(bundle,'bundle');calls.append('fit')
                (output/'validation.json').write_text('{}');(output/'recipe_lock.json').write_text('{}')
                return {'checkpoint_sha256':'f'*64}
            def numerical():
                calls.append('numerical-imports');return types.SimpleNamespace(load_approved_data=load,admit_teacher_bundle=admit),types.SimpleNamespace(fit_admitted_teacher=fit)
            original=e.preliminary_teacher
            def preliminary(*a):calls.append('preliminary-gate');return original(*a)
            with patch.object(e,'bootstrap',boot),patch.object(e,'numerical_modules',numerical),patch.object(e,'preliminary_teacher',preliminary),patch.object(e.signal,'setitimer'):
                result=e.execute(*args)
            self.assertEqual(calls,['bootstrap','preliminary-gate','numerical-imports','load-calibration','admission','fit','close'])
            self.assertEqual(result['status'],'complete');self.assertFalse(result['official_test_decoded'])
            self.assertEqual(len(list(Path(run['output']).glob('exit_*.json'))),1)
            with self.assertRaises(FileExistsError):e.execute(*args)

    def test_failed_teacher_before_numerical_or_data_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,run=self.fixture(Path(tmp),correct=4249);closed=[]
            with patch.object(e,'bootstrap',return_value=({},types.SimpleNamespace(close=lambda:closed.append(True)))),patch.object(e,'numerical_modules',side_effect=AssertionError('No numerical imports')),patch.object(e.signal,'setitimer'):
                with self.assertRaises(ValueError):e.execute(*args)
            self.assertEqual(closed,[True])
            receipt=json.loads(next(Path(run['output']).glob('exit_*.json')).read_text())
            self.assertEqual(receipt['status'],'failed');self.assertEqual(receipt['exit_code'],1)

    def test_source_slot_resume_hash_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,run=self.fixture(Path(tmp))
            with self.assertRaises(ValueError):e.verify_plan(args[0],'0'*64,*args[2:],False)
            with self.assertRaises(ValueError):e.verify_plan(*args,True)
            slot=json.loads(args[2].read_text());slot['other_science_threads']=8
            args[2].write_text(json.dumps(slot));changed=(args[0],args[1],args[2],e.digest(args[2].read_bytes()),args[4])
            with self.assertRaises(ValueError):e.verify_plan(*changed,False)
            self.assertFalse(Path(run['output']).exists())

if __name__=='__main__':unittest.main()
