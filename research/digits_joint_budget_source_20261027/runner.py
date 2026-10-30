#!/usr/bin/env python3
"""Portable fixed seven-budget J/C grid. Fresh approved matched parent only.

Import configures numerical libraries but does not load real datasets/artifacts,
start processes, fit, infer, or create a registry. Science requires a NEW release.
"""
from __future__ import annotations
import importlib.util,pathlib,hashlib,json,ast,math,os,sys,time,signal,subprocess,argparse,resource,datetime
import grid_contract as contract
import parent_contract
HERE=pathlib.Path(__file__).resolve().parent
MATCHED,DEPENDENCY=parent_contract.dependency(HERE)
HELPER=MATCHED/'runner.py'
# The matched runner uses its own top-level portable_contract module. Bind that
# exact verified module without changing sys.path or accepting a shadow module.
def _load_matched():
 name='portable_contract';path=MATCHED/'portable_contract.py'
 if name in sys.modules:
  if pathlib.Path(sys.modules[name].__file__).resolve()!=path.resolve():raise RuntimeError('shadowed matched portable_contract')
 else:
  spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
 spec=importlib.util.spec_from_file_location('canaria_matched_helper',HELPER);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
m=_load_matched()
np=m.np;torch=m.torch;nn=m.nn
for name in ('dump','now','tensor_hash','state_hash','save_state','freeze','sqrt64','fit_arm','feature_metrics','logits_from_endpoint','assert_environment','tree_bytes','rss_bytes','claim_attempt'):globals()[name]=getattr(m,name)
REGISTRY=DEADLINE=WORKSPACE=REPLICATION_ID=None
PARENT=PARENT_WORKSPACE=PARENT_MANIFEST_PATH=PARENT_MANIFEST=None
WIDTHS=(2,4,6,8,12,16,24);NEW_WIDTHS=(2,4,6,12,16,24);BUDGETS=tuple(256*h for h in WIDTHS)
def manifest_hashes():return contract.manifest_hashes(HERE)
def source_identity():return contract.package_identity(HERE)
def configure_parent_context(workspace,manifest_path):
 global PARENT,PARENT_WORKSPACE,PARENT_MANIFEST_PATH,PARENT_MANIFEST
 PARENT_WORKSPACE=pathlib.Path(workspace).resolve();PARENT_MANIFEST_PATH=pathlib.Path(manifest_path).resolve()
 PARENT_MANIFEST=json.loads(PARENT_MANIFEST_PATH.read_text())
 PARENT=parent_contract.verify_parent(PARENT_WORKSPACE,PARENT_MANIFEST,HERE)
 return PARENT

def verify_parent():
 if PARENT_WORKSPACE is None or PARENT_MANIFEST is None:raise ValueError('completed fresh matched parent not configured')
 if json.loads(PARENT_MANIFEST_PATH.read_text())!=PARENT_MANIFEST:raise ValueError('parent descriptor changed')
 root=parent_contract.verify_parent(PARENT_WORKSPACE,PARENT_MANIFEST,HERE)
 if root!=PARENT:raise ValueError('parent root changed')
 return True

def validate_release(release):
 contract.validate_release(release,HERE)
 verify_parent()
 if release['parent_manifest_sha256']!=sha(PARENT_MANIFEST_PATH) or release['parent_replication_id']!=PARENT_MANIFEST['parent_replication_id'] or release['matched_package_sha256']!=PARENT_MANIFEST['matched_package_sha256']:raise ValueError('release parent binding mismatch')
 return release

def configure_context(workspace,release):
 global REGISTRY,DEADLINE,WORKSPACE,REPLICATION_ID
 REGISTRY,out=contract.context(workspace,release['replication_id'],HERE)
 for path in (REGISTRY,out):
  if path.is_relative_to(PARENT_WORKSPACE):raise ValueError('grid custody workspace must be separate from parent workspace')
 DEADLINE=contract.deadline_timestamp(release['deadline_utc'])
 WORKSPACE=pathlib.Path(workspace).resolve();REPLICATION_ID=release['replication_id']
 return out



def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def load_npz(path):
 with np.load(path,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}

def load_state(module,path):
 arrays=load_npz(path);expected=module.state_dict()
 if set(arrays)!=set(expected):raise RuntimeError('state keys changed')
 for k,t in expected.items():
  a=arrays[k]
  if a.shape!=tuple(t.shape) or a.dtype!=t.numpy().dtype or not np.isfinite(a).all():raise RuntimeError('state type/shape/nonfinite '+k)
 module.load_state_dict({k:torch.from_numpy(v) for k,v in arrays.items()},strict=True);return module

def make_student(arm,h,seeds):
 if h not in WIDTHS:raise ValueError('unsupported width')
 if arm=='J':return nn.ModuleList([m.TinyRes(64,h,seeds['j1']),m.TinyRes(64,h,seeds['j2'])])
 if arm=='C':return m.TinyRes(64,2*h,seeds['c'])
 raise ValueError(arm)

def reused_cells(old):return [dict(old['arms'][a],budget=2048,arm=a,reused=True) for a in ('J','C')]

def run_seed(seed,out):
 out.mkdir(exist_ok=False);start=time.monotonic();dump(out/'START.json',{'id':seed['id'],'seed':seed,'utc':now(),'package_sha256':source_identity(),'files':manifest_hashes()})
 def timeout(*_):raise TimeoutError('teacher subprocess600second cap')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
 try:
  verify_parent();env=assert_environment();m.verify_sources();dump(out/'environment.json',env)
  base=PARENT/f"seed_{seed['id']}";old=json.loads((base/'result.json').read_text())
  parent_seed=next(s for s in json.loads((MATCHED/'SEEDS.json').read_text())['seeds'] if s['id']==seed['id'])
  parent_contract.validate_reused_parent_row(old,parent_seed)
  original_summary=json.loads((PARENT/'SUMMARY.json').read_text())
  if next(r for r in original_summary['all_rows'] if r['id']==seed['id'])!=old:raise RuntimeError('parent summary mismatch')
  teacher=m.Net(old['seed']['teacher']);load_state(teacher,base/'teacher_initial.npz')
  if state_hash(teacher)!=old['teacher_initial_sha256']:raise RuntimeError('parent initial teacher identity')
  load_state(teacher,base/'teacher_final.npz');freeze(teacher);teacher_hash=state_hash(teacher)
  if teacher_hash!=old['teacher_final_sha256']:raise RuntimeError('parent final teacher identity')
  data=m.load_partition();saved_hash=json.loads((base/'data_hashes.json').read_text())
  if {k:tensor_hash(data[k]) for k in ('Xt','yt','Xv','yv')}!=saved_hash:raise RuntimeError('parent data mismatch')
  split=load_npz(base/'split_ids.npz')
  if not np.array_equal(split['train'],data['train_ids']) or not np.array_equal(split['validation'],data['validation_ids']):raise RuntimeError('split mismatch')
  targets=load_npz(base/'fit_targets.npz')
  if set(targets)!= {'a0','a1','a2'} or any(v.shape!=(1077,64) or v.dtype!=np.float32 or not np.isfinite(v).all() for v in targets.values()):raise RuntimeError('invalid fit targets')
  a=[torch.from_numpy(targets[k]) for k in ('a0','a1','a2')]
  sched=torch.from_numpy(np.load(base/'schedule.npy',allow_pickle=False))
  if sched.dtype!=torch.int64 or sched.shape!=(600,128) or int(sched.min())<0 or int(sched.max())>=1077 or tensor_hash(sched)!=old['schedule_sha256']:raise RuntimeError('schedule mismatch')
  models={};fits={}
  with sqrt64() as counts:
   for width in seed['widths']:
    h=width['h']
    if h not in NEW_WIDTHS or width['budget']!=256*h:raise RuntimeError('nonadmitted width')
    for arm in ('J','C'):
     key=f'{arm}_{256*h}';module=make_student(arm,h,width)
     if sum(p.numel() for p in module.parameters())!=256*h:raise RuntimeError('budget mismatch')
     save_state(out/f'{key}_initial.npz',module);initial=state_hash(module)
     dump(out/f'{key}_START.json',{'arm':arm,'budget':256*h,'utc':now(),'initial_state_sha256':initial,'schedule_sha256':tensor_hash(sched)})
     fits[key]=fit_arm(arm,module,a,sched,out/f'{key}_updates.jsonl');save_state(out/f'{key}_final.npz',module);models[key]=module
     if state_hash(teacher)!=teacher_hash or any(p.grad is not None for p in teacher.parameters()):raise RuntimeError('teacher mutated')
  if set(models)!={f'{arm}_{256*h}' for arm in ('J','C') for h in NEW_WIDTHS}:raise RuntimeError('fit census mismatch')
  # No validation forward until all twelve fits complete; existing2048 scalar reuse is not fitting.
  with torch.no_grad():
   tl,va=teacher(data['Xv'],return_acts=True);target=va[2];saved=load_npz(base/'validation_arrays.npz')
   if not np.array_equal(target.numpy(),saved['target']) or not np.array_equal(tl.numpy(),saved['teacher_logits']) or not np.array_equal(data['yv'].numpy(),saved['labels']):raise RuntimeError('parent validation identity mismatch')
   tc=int((tl.argmax(1)==data['yv']).sum())
   if tc!=old['teacher_correct']:raise RuntimeError('teacher count mismatch')
   cells=reused_cells(old);arrays={'target':target.numpy(),'teacher_logits':tl.numpy(),'labels':data['yv'].numpy()}
   for key,module in models.items():
    arm,b=key.split('_');h=module[1](module[0](va[0])) if arm=='J' else module(va[0]);logits=logits_from_endpoint(teacher,h)
    if not torch.isfinite(h).all() or not torch.isfinite(logits).all():raise RuntimeError('nonfinite evaluation')
    correct=int((logits.argmax(1)==data['yv']).sum());row=feature_metrics(h,target)
    row.update(budget=int(b),arm=arm,reused=False,correct=correct,n=270,accuracy=correct/270,quality_pass=row['nmse']<=.08 and correct/270>=tc/270-.02,state_sha256=state_hash(module));cells.append(row);arrays[key+'_endpoint']=h.numpy();arrays[key+'_logits']=logits.numpy()
   np.savez_compressed(out/'validation_arrays.npz',**arrays)
  verify_parent();result={'id':seed['id'],'status':'COMPLETE','parent_result_sha256':sha(base/'result.json'),'parent_teacher_initial_sha256':old['teacher_initial_sha256'],'parent_teacher_final_sha256':teacher_hash,'schedule_sha256':tensor_hash(sched),'teacher_correct':tc,'cells':cells,'fit_records':fits,'sqrt_counts':counts,'elapsed_seconds':time.monotonic()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'finished_utc':now()}
  dump(out/'result.json',result);dump(out/'FILES_SHA256.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()});return result
 except BaseException as e:
  dump(out/'FAILURE.json',{'id':seed['id'],'status':'FAILED','exception':type(e).__name__,'reason':str(e),'utc':now(),'elapsed_seconds':time.monotonic()-start});raise
 finally:signal.alarm(0)

def analyze(rows):
 incomplete={'decision':'INCOMPLETE','no_exclusion_or_replacement':True}
 if len(rows)!=16 or {r.get('id') for r in rows}!=set(range(16)) or any(r.get('status')!='COMPLETE' for r in rows):return incomplete
 minima=[];pass_counts={a:{str(b):0 for b in BUDGETS} for a in ('C','J')};patterns=[]
 for r in sorted(rows,key=lambda r:r['id']):
  cells=r.get('cells',[])
  if len(cells)!=14 or {(c['arm'],c['budget']) for c in cells}!={(a,b) for a in ('C','J') for b in BUDGETS}:return incomplete
  for c in cells:
   if not math.isfinite(c['nmse']) or c['nmse']<0 or c['n']!=270 or not isinstance(c['correct'],int) or not 0<=c['correct']<=270 or c['reused']!=(c['budget']==2048):return incomplete
   actual=c['nmse']<=.08 and c['correct']/270>=r['teacher_correct']/270-.02
   if c['quality_pass']!=actual:return incomplete
   pass_counts[c['arm']][str(c['budget'])]+=int(actual)
  row={'id':r['id']}
  for arm in ('C','J'):
   by={c['budget']:c['quality_pass'] for c in cells if c['arm']==arm};passing=[b for b in BUDGETS if by[b]];row[arm]=min(passing) if passing else None
   patterns.append({'id':r['id'],'arm':arm,'pass_by_budget':[by[b] for b in BUDGETS],'nonmonotone':any(by[BUDGETS[i]] and not by[BUDGETS[j]] for i in range(7) for j in range(i+1,7))})
  minima.append(row)
 coverage={'both':sum(x['C'] is not None and x['J'] is not None for x in minima),'C_only':sum(x['C'] is not None and x['J'] is None for x in minima),'J_only':sum(x['C'] is None and x['J'] is not None for x in minima),'neither':sum(x['C'] is None and x['J'] is None for x in minima)}
 out={'decision':'CENSORED_NO_COHORT_RATIO','all_rows':rows,'minima':minima,'coverage':coverage,'pass_counts':pass_counts,'patterns':patterns,'scope':'exploratory conditional control-grid extension; reused digits/teachers;2048 cells reused; no fresh-data or objective-only claim'}
 if coverage['both']==16:
  v=np.array([math.log2(x['C']/x['J']) for x in minima]);seed=json.loads((HERE/'SEEDS.json').read_text())['bootstrap_seed'];ix=np.random.default_rng(seed).integers(0,16,(100000,16));ci=np.quantile(v[ix].mean(1),[.025,.975]);out.update(decision='CONDITIONAL_GRID_ADVANTAGE' if ci[1]<0 else 'CONDITIONAL_GRID_DISADVANTAGE' if ci[0]>0 else 'UNCERTAIN',mean_log2_C_over_J=float(v.mean()),ci95_log2=ci.tolist(),geometric_ratio=float(2**v.mean()),ratio_ci95=(2**ci).tolist(),bootstrap_seed=seed,bootstrap_draws=100000)
 return out

def cohort(release_path,out):
 release=validate_release(json.loads(release_path.read_text()))
 if REGISTRY is None or DEADLINE is None or WORKSPACE is None:raise RuntimeError('custody context not configured')
 expected_registry,expected_out=contract.context(WORKSPACE,release['replication_id'],HERE)
 if REGISTRY!=expected_registry or out.resolve()!=expected_out:raise RuntimeError('custody/output binding mismatch')
 if DEADLINE!=contract.deadline_timestamp(release['deadline_utc']):raise RuntimeError('deadline binding mismatch')
 # Atomic permanent attempt registry. No automatic retry even if no outcome.
 claim_attempt(REGISTRY,{'utc':now(),'package_sha256':source_identity(),'replication_id':REPLICATION_ID,'output_relative':str(out.relative_to(WORKSPACE))});out.mkdir(exist_ok=False,parents=True)
 token=os.urandom(32).hex();token_hash=hashlib.sha256(token.encode()).hexdigest();start=time.time();end=min(start+10800,DEADLINE)
 dump(REGISTRY/'cohort.json',{'status':'STARTED','utc':now(),'pid':os.getpid(),'release':release,'token_hash':token_hash,'out':str(out.resolve()),'end_timestamp':end})
 dump(out/'run_contract.json',contract.public_release(release));seeds=json.loads((HERE/'SEEDS.json').read_text())['seeds'];pending=list(seeds);running={};records=[];abort=None
 try:
  while pending or running:
   while pending and len(running)<2 and abort is None:
    s=pending.pop(0);dump(REGISTRY/f"seed_{s['id']}.started.json",{'id':s['id'],'utc':now()});log=None;st=time.time()
    try:
     log=(out/f"seed_{s['id']}.log").open('x')
     env=os.environ.copy();env['CANARIA_GRID_TOKEN']=token
     cmd=[sys.executable,str(__file__),'seed','--id',str(s['id']),'--workspace',str(WORKSPACE),'--replication-id',REPLICATION_ID,'--parent-workspace',str(PARENT_WORKSPACE),'--parent-manifest',str(PARENT_MANIFEST_PATH)]
     p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env);running[p.pid]=(p,log,s,st)
    except BaseException as e:
     if log is not None:log.close()
     abort=abort or 'spawn '+type(e).__name__+': '+str(e)
     record={'id':s['id'],'status':'SPAWN_FAILED','returncode':None,'reason':abort,'elapsed_seconds':time.time()-st}
     records.append(record);dump(out/f"seed_{s['id']}_EXIT.json",record);break
   if time.time()>=end:abort=abort or 'cohort deadline/cap'
   if tree_bytes(out)>1024**3:abort=abort or 'aggregate evidence cap'
   for pid,(p,log,s,st) in list(running.items()):
    reason=abort
    if time.time()-st>=600:reason=reason or 'seed wall cap'
    if rss_bytes(pid)>4*1024**3:reason=reason or 'seed RSS cap'
    if reason:abort=abort or reason
    rc=p.poll()
    if rc is not None:
     log.close();record={'id':s['id'],'returncode':rc,'reason':reason,'elapsed_seconds':time.time()-st};records.append(record);dump(out/f"seed_{s['id']}_EXIT.json",record);del running[pid]
     if rc!=0:abort=abort or 'seed failed'
   if abort:break  # Enter bounded TERM/KILL cleanup immediately; no cap polling loop.
   time.sleep(.2)
 except BaseException as e:
  abort=abort or 'coordinator '+type(e).__name__+': '+str(e)
 finally:
  # A coordinator failure/cap may not leave orphan scientific processes.
  for pid,(p,log,seed,st) in list(running.items()):
   if p.poll() is None:p.terminate()
   try:p.wait(timeout=5)
   except subprocess.TimeoutExpired:p.kill();p.wait()
   log.close();record={'id':seed['id'],'returncode':p.returncode,'reason':abort or 'coordinator cleanup','elapsed_seconds':time.time()-st};records.append(record)
   path=out/f"seed_{seed['id']}_EXIT.json"
   if not path.exists():dump(path,record)
   del running[pid]
 rows=[]
 for s in seeds:
  path=out/f"seed_{s['id']}"/'result.json'
  if path.exists():rows.append(json.loads(path.read_text()))
 summary=analyze(rows)
 if abort is not None or len(records)!=16 or any(r['returncode']!=0 for r in records):summary={'decision':'INCOMPLETE','reason':abort or 'incomplete successful exits','completed_results':len(rows),'no_exclusion_or_replacement':True}
 summary.update(attempts=records,not_started=[s['id'] for s in pending],abort_reason=abort,finished_utc=now());dump(out/'SUMMARY.json',summary);dump(REGISTRY/'FINISHED.json',{'utc':now(),'decision':summary['decision'],'summary_sha256':sha(out/'SUMMARY.json')});dump(out/'FILES_SHA256.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
 return summary

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('mode',choices=('cohort','seed'))
 ap.add_argument('--workspace',type=pathlib.Path,required=True)
 ap.add_argument('--release',type=pathlib.Path)
 ap.add_argument('--parent-workspace',type=pathlib.Path,required=True)
 ap.add_argument('--parent-manifest',type=pathlib.Path,required=True)
 ap.add_argument('--replication-id')
 ap.add_argument('--id',type=int)
 a=ap.parse_args()
 configure_parent_context(a.parent_workspace,a.parent_manifest)
 if a.mode=='cohort':
  if a.release is None:ap.error('reviewed new-replication release required')
  release=validate_release(json.loads(a.release.read_text()))
  out=configure_context(a.workspace,release)
  summary=cohort(a.release,out);print(json.dumps({k:v for k,v in summary.items() if k!='all_rows'}))
  if summary['decision']=='INCOMPLETE':raise SystemExit(2)
  return
 if a.replication_id is None or a.id is None:ap.error('internal seed identity required')
 registry,_=contract.context(a.workspace,a.replication_id,HERE)
 r=json.loads((registry/'cohort.json').read_text());token=os.environ.pop('CANARIA_GRID_TOKEN','')
 if not token or hashlib.sha256(token.encode()).hexdigest()!=r['token_hash'] or time.time()>=r['end_timestamp']:raise RuntimeError('invalid internal seed invocation')
 release=validate_release(r['release'])
 if release['replication_id']!=a.replication_id:raise RuntimeError('replication identity mismatch')
 out=configure_context(a.workspace,release)
 if out!=pathlib.Path(r['out']):raise RuntimeError('internal output mismatch')
 if not (REGISTRY/f'seed_{a.id}.started.json').exists():raise RuntimeError('seed not admitted')
 seed=next(s for s in json.loads((HERE/'SEEDS.json').read_text())['seeds'] if s['id']==a.id)
 run_seed(seed,out/f'seed_{a.id}')

if __name__=='__main__':main()
