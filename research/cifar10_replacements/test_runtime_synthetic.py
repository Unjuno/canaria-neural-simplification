"""Synthetic tensors only. Temporary artifacts contain no real dataset/checkpoint."""
import copy
import datetime
import tempfile
import time
import random
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
import numpy as np
from . import core as c
from . import runtime as r

torch.set_num_threads(1)

class TinyTeacher:
    def features(self,x):
        return x.detach(),(x+.1).detach(),(x+.2).detach()

class RuntimeTests(unittest.TestCase):
    def session(self,launch=None):
        launch=launch or dict(started=time.time(),expiry=time.time()+7200,approved_plan_sha256='a'*64)
        return r.FitSession(TinyTeacher(),'b'*64,tuple(range(5000)),c.SEEDS[0],launch,{},'synthetic')

    def test_centered_nmse_full_dataset(self):
        y=torch.tensor([1.,2.,4.,8.]).reshape(4,1,1,1)
        p=y+2
        allstats=c.EndpointStats(); allstats.update(p,y)
        batched=c.EndpointStats(); batched.update(p[:1],y[:1]); batched.update(p[1:],y[1:])
        self.assertEqual(allstats.result(),batched.result())
        expected=float(((p-y).double()**2).sum()/(((y.double()-y.double().mean(0))**2).sum()+4e-12))
        self.assertEqual(allstats.result()['centered_span_nmse'],expected)
        const=c.EndpointStats(); const.update(torch.ones(2,1,1,1)*2,torch.ones(2,1,1,1))
        self.assertEqual(const.result()['centered_span_nmse'],1e12)

    def test_resume_exact_and_schedule_guard(self):
        with tempfile.TemporaryDirectory() as d:
            launch=r.open_launch(d,'a'*64)
            a=self.session(launch); x=torch.rand(128,32,2,2)
            with self.assertRaises(ValueError): a.step((1,2),x)
            a.step(a.batches[0],x); path=Path(d)/'last.ckpt'; a.save(path)
            expected_torch=torch.rand(2); expected_python=random.random(); expected_numpy=np.random.rand(2)
            b=self.session(launch); b.restore(path)
            self.assertTrue(torch.equal(torch.rand(2),expected_torch))
            self.assertEqual(random.random(),expected_python)
            self.assertTrue(np.array_equal(np.random.rand(2),expected_numpy))
            self.assertEqual(a.cursor,b.cursor)
            a.step(a.batches[1],x); b.step(b.batches[1],x)
            for n in a.arms:
                self.assertEqual(r.tensor_hash(a.arms[n].state_dict()),r.tensor_hash(b.arms[n].state_dict()))
            self.assertEqual(a.lock_sha256,b.lock_sha256)
            changed=self.session(dict(launch,started=launch['started']+1))
            with self.assertRaises(ValueError):changed.restore(path)
            bad=r.decode(path.read_bytes()); bad['cursor']=1001
            r.atomic_bytes(path,r.envelope(bad))
            with self.assertRaises(ValueError): b.restore(path)

    def test_production_guards_before_work(self):
        from .launch import bootstrap
        with self.assertRaises(RuntimeError):bootstrap('/never-open','a'*64)
        with self.assertRaises(PermissionError):
            r.fit_admitted_teacher(None,None,'/never-open',{},None,worker_lock=None)
        with self.assertRaises(ValueError):
            r.fit_admitted_teacher(None,None,'/never-open',{},None,worker_lock=None,execution_authorized=True)

    def test_original_cap_and_expired_resume_before_read(self):
        self.assertEqual(datetime.datetime.fromtimestamp(r.DEADLINE,datetime.timezone.utc).isoformat(),'2026-10-29T06:50:00+00:00')
        with tempfile.TemporaryDirectory() as d:
            launch=r.open_launch(d,'a'*64,now=100.)
            self.assertEqual(r.open_launch(d,'a'*64,resume=True,now=101.),launch)
            with self.assertRaises(FileExistsError):r.open_launch(d,'a'*64,now=102.)
            with self.assertRaises(TimeoutError):r.open_launch(d,'a'*64,resume=True,now=7300.)
        a=self.session(); a.launch['expiry']=0
        with patch.object(Path,'read_bytes',side_effect=AssertionError('Must not read')):
            with self.assertRaises(TimeoutError):a.restore('/not-real')

    def test_nonfinite_never_overwrites_last_good(self):
        with tempfile.TemporaryDirectory() as d:
            a=self.session(); path=Path(d)/'last.ckpt'; a.save(path); before=path.read_bytes()
            optimizer=a.optimizers['Comp-joint']; original=optimizer.step
            def corrupt():
                original()
                with torch.no_grad():next(a.arms['Comp-joint'].parameters()).fill_(float('inf'))
            with patch.object(optimizer,'step',corrupt):
                with self.assertRaises(FloatingPointError):a.step(a.batches[0],torch.rand(128,32,2,2))
            self.assertEqual(a.cursor,0)
            with self.assertRaises(RuntimeError):a.save(path)
            self.assertEqual(path.read_bytes(),before)
            a.restore(path); self.assertFalse(a.failed)

    def test_final_validation_accepted_with_exact_ids(self):
        a=self.session(); a.cursor=1000; a.teacher_validation_correct=4250
        ids=tuple(range(5000,10000))
        a.lock['data_identity']={'splits':{'validation':{'sha256':c.sha(np.asarray(ids,dtype='<i8').tobytes())}}}
        a.arms={n:torch.nn.Identity() for n in a.arms}
        a.teacher.installed_logits=lambda x,arm:torch.zeros(len(x),10)
        result=a.final_validation([(ids,torch.ones(5000,32,1,1),torch.zeros(5000,dtype=torch.long))],ids)
        self.assertTrue(all(v['correct']==5000 and v['utility_drop_le_2pp'] for v in result['arms'].values()))
        self.assertFalse(result['official_test_decoded'])

    def test_final_validation_and_cohort_gate(self):
        a=self.session()
        with self.assertRaises(ValueError):a.final_validation([],range(5000,10000))
        a.cursor=1000
        with self.assertRaises(ValueError):a.final_validation([],range(5000))
        records={seed:dict(status='teacher-quality-failed') for seed in c.SEEDS}
        with self.assertRaises(ValueError):r.cohort_release_gate(records,{})
        approval=dict(action='separately-reviewed-official-test-release',cohort_sha256=c.canonical_hash(records))
        self.assertFalse(r.cohort_release_gate(records,approval)['test_evaluation_implemented'])
        records[c.SEEDS[0]]={'status':'complete'}
        with self.assertRaises(ValueError):r.cohort_release_gate(records,approval)

if __name__=='__main__':unittest.main()
