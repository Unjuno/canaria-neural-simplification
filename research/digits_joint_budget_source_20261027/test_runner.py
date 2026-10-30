"""Synthetic numerical checks. No dataset access or real model loading."""
import pathlib,unittest,importlib.util,tempfile,json,copy
from unittest.mock import patch
from test_parent_contract import release as synthetic_release
P=pathlib.Path(__file__).with_name('runner.py')
s=importlib.util.spec_from_file_location('grid',P);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class GridTests(unittest.TestCase):
 def setUp(self):self.assertIsNotNone(g,'restricted grid implementation absent')
 def rows(self):return [{'id':i,'status':'COMPLETE','cells':[{'budget':b,'arm':a,'nmse':.01,'correct':265,'n':270,'quality_pass':True,'reused':b==2048} for b in g.BUDGETS for a in ('J','C')],'teacher_correct':265} for i in range(16)]
 def test_exact_budget_parameter_counts(self):
  for h in g.WIDTHS:
   for a in ('J','C'):self.assertEqual(sum(p.numel() for p in g.make_student(a,h,{'j1':1,'j2':2,'c':3}).parameters()),256*h)
 def test_no_2048_fitting(self):self.assertEqual(g.NEW_WIDTHS,(2,4,6,12,16,24));self.assertEqual(16*len(g.NEW_WIDTHS)*2,192)
 def test_complete_ratio(self):self.assertEqual(g.analyze(self.rows())['decision'],'UNCERTAIN')
 def test_missing_cell_incomplete(self):
  r=self.rows();r[0]['cells'].pop();self.assertEqual(g.analyze(r)['decision'],'INCOMPLETE')
 def test_duplicate_cell_incomplete(self):
  r=self.rows();r[0]['cells'][-1]=copy.deepcopy(r[0]['cells'][0]);self.assertEqual(g.analyze(r)['decision'],'INCOMPLETE')
 def test_censor_no_complete_case_ratio(self):
  r=self.rows()
  for c in r[0]['cells']:
   if c['arm']=='J':c.update(nmse=.2,quality_pass=False)
  o=g.analyze(r);self.assertEqual(o['decision'],'CENSORED_NO_COHORT_RATIO');self.assertNotIn('geometric_ratio',o);self.assertEqual(o['coverage']['C_only'],1)
 def test_nonmonotone_minimum(self):
  r=self.rows()
  for row in r:
   for c in row['cells']:
    if c['budget'] in (512,1536):c.update(nmse=.2,quality_pass=False)
  o=g.analyze(r);self.assertEqual(o['minima'][0]['C'],1024)
 def test_stored_pass_not_trusted(self):
  r=self.rows();r[0]['cells'][0]['quality_pass']=False;self.assertEqual(g.analyze(r)['decision'],'INCOMPLETE')
 def test_reuse_exact_metric(self):
  old={'arms':{'J':{'nmse':.03,'correct':260,'n':270,'accuracy':260/270,'quality_pass':True,'state_sha256':'x'},'C':{'nmse':.02,'correct':261,'n':270,'accuracy':261/270,'quality_pass':True,'state_sha256':'y'}}}
  r=g.reused_cells(old);self.assertTrue(all(x['reused'] for x in r));self.assertEqual({k:r[0][k] for k in old['arms']['J']},old['arms']['J'])
 def test_npz_pickle_refused(self):
  import numpy as np
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'x.npz';np.savez(p,x=np.array([{}],dtype=object))
   with self.assertRaises(ValueError):g.load_npz(p)
 def test_release_tamper(self):
  with self.assertRaises(ValueError):g.validate_release({'parent_approved':True,'independent_review':'GO','implementation_commit':'wrong','files':{}})
 def test_state_shape_dtype_rejected(self):
  import numpy as np
  module=g.make_student('C',2,{'c':1})
  arrays={k:v.numpy().copy() for k,v in module.state_dict().items()};k=next(iter(arrays));arrays[k]=arrays[k].astype(np.float64)
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'state.npz';np.savez(p,**arrays)
   with self.assertRaisesRegex(RuntimeError,'state type'):g.load_state(module,p)
 def test_synthetic_fit_shared_schedule_input_immutable(self):
  t=g.torch;rng=t.Generator().manual_seed(54);a=[t.randn(32,64,generator=rng) for _ in range(3)];schedule=g.m.schedule(32,99,steps=3);h=g.tensor_hash(schedule);before=[g.tensor_hash(v) for v in a]
  C=g.make_student('C',2,{'c':1});J=g.make_student('J',2,{'j1':2,'j2':3});ch=g.state_hash(C)
  with g.sqrt64():r=g.fit_arm('J',J,a,schedule)
  self.assertEqual(g.state_hash(C),ch);self.assertEqual(r['parameter_steps'],[3]*4);self.assertEqual(g.tensor_hash(schedule),h);self.assertEqual(before,[g.tensor_hash(v) for v in a])
 def test_no_teacher_training_or_old_student_load(self):
  import ast,inspect
  calls=[ast.unparse(n.func) for n in ast.walk(ast.parse(inspect.getsource(g.run_seed))) if isinstance(n,ast.Call)]
  self.assertFalse(any('train_model' in x for x in calls));self.assertNotIn('m.initialize',calls)
 def test_release_rechecks_parent_before_attempt(self):
  with patch.object(g,'verify_parent',side_effect=ValueError('changed parent')):
   with self.assertRaisesRegex(ValueError,'changed parent'):g.validate_release(synthetic_release())
 def test_import_does_not_configure_an_attempt(self):
  self.assertIsNone(g.REGISTRY);self.assertIsNone(g.PARENT);self.assertIsNone(g.DEADLINE)
 def test_nonmonotonicity_is_reported_not_repaired(self):
  rows=self.rows()
  rows[0]['cells'][2].update(nmse=.2,quality_pass=False)
  report=g.analyze(rows)
  self.assertTrue(next(p for p in report['patterns'] if p['id']==0 and p['arm']=='J')['nonmonotone'])
  self.assertEqual(report['minima'][0]['J'],512)
 def test_reused_2048_flag_required(self):
  rows=self.rows()
  next(c for c in rows[0]['cells'] if c['budget']==2048)['reused']=False
  self.assertEqual(g.analyze(rows)['decision'],'INCOMPLETE')
 def test_attempt_atomic_and_no_retry(self):
  with tempfile.TemporaryDirectory() as d:
   registry=pathlib.Path(d)/'registry';g.claim_attempt(registry,{'synthetic':True})
   with self.assertRaises(FileExistsError):g.claim_attempt(registry,{'synthetic':True})

 def test_cap_escalates_and_preserves_first_reason(self):
  import tempfile
  from unittest.mock import patch
  m=g;events=[];logs=[]
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
  m=g;logs=[]
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
if __name__=='__main__':unittest.main(verbosity=2)
