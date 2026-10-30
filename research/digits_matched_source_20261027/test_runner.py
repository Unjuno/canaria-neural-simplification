"""Synthetic-only preparation checks. No sklearn digits loading or teacher fitting."""
import copy, importlib.util, json, os, pathlib, sys, unittest
from unittest.mock import patch
P=pathlib.Path(__file__).parent
sys.path.insert(0,str(P))
class MatchedTests(unittest.TestCase):
 def setUp(self):
  self.assertTrue((P/'runner.py').exists(), 'restricted runner is not implemented')
  spec=importlib.util.spec_from_file_location('matched',P/'runner.py');self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
  self.t=self.m.torch
 def fixture(self):
  t=self.t;g=t.Generator().manual_seed(11)
  return [t.randn(32,64,generator=g) for _ in range(3)]
 def test_exact_clone_and_no_alias(self):
  m=self.m;L,J,C=m.initialize({'pair1':1,'pair2':2,'comp':3})
  self.assertEqual(m.state_hash(L),m.state_hash(J));self.assertEqual(sum(p.numel() for p in L.parameters()),2048);self.assertEqual(sum(p.numel() for p in C.parameters()),2048)
  self.assertTrue(all(a.data_ptr()!=b.data_ptr() for a,b in zip(L.parameters(),J.parameters())))
 def test_shared_schedule_is_immutable_and_local_disjoint_equivalent(self):
  m=self.m;t=self.t;L,J,C=m.initialize({'pair1':1,'pair2':2,'comp':3});a=self.fixture();sch=m.schedule(32,77,steps=5);before=m.tensor_hash(sch);ref=copy.deepcopy(L)
  opts=[m.optimizer(x.parameters()) for x in ref]
  with m.sqrt64():
   for ix in sch:
    for k in range(2):
     opts[k].zero_grad(set_to_none=True);t.nn.functional.mse_loss(ref[k](a[k][ix]),a[k+1][ix]).backward();opts[k].step()
   out=m.fit_arm('L',L,a,sch)
  self.assertEqual(m.state_hash(L),m.state_hash(ref));self.assertEqual(before,m.tensor_hash(sch));self.assertEqual(out['parameter_steps'],[5]*4)
 def test_joint_does_not_mutate_inputs_or_other_arm(self):
  m=self.m;L,J,C=m.initialize({'pair1':1,'pair2':2,'comp':3});a=self.fixture();before=[m.tensor_hash(x) for x in a];lh=m.state_hash(L);ch=m.state_hash(C)
  with m.sqrt64():o=m.fit_arm('J',J,a,m.schedule(32,77,steps=3))
  self.assertEqual(before,[m.tensor_hash(x) for x in a]);self.assertEqual(lh,m.state_hash(L));self.assertEqual(ch,m.state_hash(C));self.assertEqual(o['parameter_steps'],[3]*4)
 def test_frozen_teacher_unchanged(self):
  m=self.m;teacher=m.Net(5);m.freeze(teacher);before=m.state_hash(teacher)
  self.assertFalse(any(p.requires_grad for p in teacher.parameters()));x=self.fixture()[0]
  with self.t.no_grad():a=teacher(x,return_acts=True)[1][:3]
  L,_,_=m.initialize({'pair1':1,'pair2':2,'comp':3})
  with m.sqrt64():m.fit_arm('L',L,a,m.schedule(32,77,steps=2))
  self.assertEqual(before,m.state_hash(teacher));self.assertTrue(all(p.grad is None for p in teacher.parameters()))
 def test_sqrt_restored_on_exception(self):
  m=self.m;t=self.t;step=t.optim.AdamW.step;sqrt=t.Tensor.sqrt
  with self.assertRaisesRegex(ValueError,'sentinel'):
   with m.sqrt64():raise ValueError('sentinel')
  self.assertIs(step,t.optim.AdamW.step);self.assertIs(sqrt,t.Tensor.sqrt)
 def test_only_train_validation_arrays(self):
  m=self.m;np=m.np;x=np.arange(1797*64,dtype=np.float32).reshape(1797,64);y=np.arange(1797)%10
  data=m.partition_arrays(x,y)
  self.assertEqual(set(data),{'Xt','yt','Xv','yv','train_ids','validation_ids'})
  self.assertEqual(data['Xt'].shape,(1077,64));self.assertEqual(data['Xv'].shape,(270,64));self.assertFalse(set(data['train_ids'])&set(data['validation_ids']))
 def test_source_hash_and_extraction_allowlist(self):
  self.m.verify_sources();self.assertEqual(set(self.m.EXTRACTED),{'ResBlock','Net','TinyRes','train_model'})
  self.assertFalse(hasattr(self.m,'run'));self.assertFalse(hasattr(self.m,'datasets'))
 def test_nonfinite_is_failure(self):
  m=self.m;a=self.fixture();a[0][0,0]=float('nan');L,_,_=m.initialize({'pair1':1,'pair2':2,'comp':3})
  sch=self.t.zeros((1,128),dtype=self.t.long)
  with self.assertRaisesRegex(RuntimeError,'nonfinite'):
   with m.sqrt64():m.fit_arm('L',L,a,sch)
 def test_release_missing_and_tampered_rejected(self):
  with self.assertRaises((ValueError,KeyError)):self.m.validate_release({})
 def test_release_tampering_rejected(self):
  from test_portable_contract import synthetic_release
  m=self.m;r=synthetic_release(P)
  r['files']['runner.py']='0'*64
  with self.assertRaisesRegex(ValueError,'mismatch'):m.validate_release(r)
 def test_exclusive_attempt_claim(self):
  import tempfile
  self.assertTrue(hasattr(self.m,'claim_attempt'),'exclusive claim helper missing')
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'attempt'
   self.m.claim_attempt(p,{'synthetic':True})
   with self.assertRaises(FileExistsError):self.m.claim_attempt(p,{'synthetic':True})
 def test_summary_primary_guardrail_and_incomplete(self):
  m=self.m
  self.assertEqual(m.analyze([])['decision'],'INCOMPLETE')
  rows=[{'id':i,'status':'COMPLETE','teacher_correct':260,'arms':{'L':{'nmse':.2,'correct':240},'J':{'nmse':.1,'correct':240},'C':{'nmse':.05,'correct':250}}} for i in range(16)]
  o=m.analyze(rows);self.assertEqual(o['feature_gate'],'PASS');self.assertEqual(o['accuracy_gate'],'PASS');self.assertEqual(o['primary_mean_log2_J_over_L'],-1)
 def test_cap_escalates_and_preserves_first_reason(self):
  import tempfile
  from unittest.mock import patch
  m=self.m;events=[];logs=[]
  class Fake:
   counter=0
   def __init__(self,*a,**kw):
    Fake.counter+=1;self.pid=Fake.counter;self.returncode=None;logs.append(kw['stdout'])
   def poll(self):return self.returncode
   def terminate(self):events.append(('terminate',self.pid))
   def kill(self):events.append(('kill',self.pid));self.returncode=-9
   def wait(self,timeout=None):
    if self.returncode is None:raise m.subprocess.TimeoutExpired('fake',timeout)
    return self.returncode
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d);r=d/'release.json';r.write_text('{}')
   with patch.object(m,'REGISTRY',d/'registry'),patch.object(m,'WORKSPACE',d),patch.object(m,'DEADLINE',m.time.time()+300),patch.object(m,'REPLICATION_ID','replication-synthetic'),patch.object(m,'validate_release',lambda r:{'replication_id':'replication-synthetic','deadline_utc':'synthetic'}),patch.object(m,'source_identity',lambda:'synthetic'),patch.object(m.contract,'context',lambda *a:(d/'registry',d/'out')),patch.object(m.contract,'public_release',lambda r:{'synthetic':True}),patch.object(m.contract,'deadline_timestamp',lambda value:m.DEADLINE),patch.object(m.subprocess,'Popen',Fake),patch.object(m,'rss_bytes',lambda pid:5*1024**3),patch.object(m.time,'sleep',side_effect=AssertionError('cap must immediately enter cleanup')):
    o=m.cohort(r,d/'out')
   self.assertEqual(o['abort_reason'],'seed RSS cap');self.assertEqual(len(o['attempts']),2)
   self.assertEqual(events,[('terminate',1),('kill',1),('terminate',2),('kill',2)])
   self.assertTrue(all(x.closed for x in logs));self.assertEqual(o['decision'],'INCOMPLETE')
 def test_spawn_failure_is_censused_and_log_closed(self):
  import tempfile
  from unittest.mock import patch
  m=self.m;logs=[]
  def fail(*a,**kw):
   logs.append(kw['stdout']);raise OSError('synthetic spawn failure')
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d);r=d/'release.json';r.write_text('{}')
   with patch.object(m,'REGISTRY',d/'registry'),patch.object(m,'WORKSPACE',d),patch.object(m,'DEADLINE',m.time.time()+300),patch.object(m,'REPLICATION_ID','replication-synthetic'),patch.object(m,'validate_release',lambda r:{'replication_id':'replication-synthetic','deadline_utc':'synthetic'}),patch.object(m,'source_identity',lambda:'synthetic'),patch.object(m.contract,'context',lambda *a:(d/'registry',d/'out')),patch.object(m.contract,'public_release',lambda r:{'synthetic':True}),patch.object(m.contract,'deadline_timestamp',lambda value:m.DEADLINE),patch.object(m.subprocess,'Popen',fail):
    o=m.cohort(r,d/'out')
   self.assertEqual(len(o['attempts']),1);self.assertEqual(o['attempts'][0]['status'],'SPAWN_FAILED')
   self.assertEqual(o['attempts'][0]['id'],0);self.assertIsNone(o['attempts'][0]['returncode'])
   self.assertTrue((d/'out'/'seed_0_EXIT.json').exists());self.assertTrue(logs[0].closed)
   self.assertEqual(o['not_started'],list(range(1,16)));self.assertEqual(o['decision'],'INCOMPLETE')
 def test_snapshot_raw_metrics(self):
  m=self.m;t=self.t;a=t.arange(8,dtype=t.float32).reshape(2,4);b=a+1
  o=m.feature_metrics(b,a);self.assertEqual(o['sse'],8.0);self.assertEqual(o['elements'],8);self.assertEqual(o['mse'],1.0)
if __name__=='__main__':unittest.main(verbosity=2)
