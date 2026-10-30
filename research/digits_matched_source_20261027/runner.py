#!/usr/bin/env python3
"""Portable matched source. New replication only; science requires its own release.

This exported lifecycle does not reopen the completed historical attempt.
Import configures numerical settings but does not load real data or train.
"""
from __future__ import annotations
import os
ENV={'ATEN_CPU_CAPABILITY':'avx2','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','MKL_DYNAMIC':'FALSE','OMP_DYNAMIC':'FALSE','MKL_CBWR':'COMPATIBLE'}
os.environ.update(ENV)
import argparse, ast, contextlib, copy, datetime, hashlib, json, math, pathlib, platform, resource, signal, subprocess, sys, threading, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import portable_contract as contract
HERE=pathlib.Path(__file__).resolve().parent
BASE=HERE/'scientific_definitions.py'
BASE_SHA='d629f501136a83e2b958d5ec61537d4719ea7fd568e54d692189947c416453f2'
# Populated only from a reviewed NEW replication contract; no historical path.
REGISTRY=None
DEADLINE=None
WORKSPACE=None
REPLICATION_ID=None
EXTRACTED=('ResBlock','Net','TinyRes','train_model')
torch.set_num_threads(1)
if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled=False
torch.use_deterministic_algorithms(True)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def verify_sources():
 if sha(BASE)!=BASE_SHA:raise RuntimeError('pinned scientific source changed')
verify_sources()
# Only class definitions and the exact teacher function are compiled. Historical
# datasets(), run(), full grid, test evaluation and __main__ are never compiled.
_tree=ast.parse(BASE.read_text());_nodes=[x for x in _tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in EXTRACTED]
if {x.name for x in _nodes}!=set(EXTRACTED):raise RuntimeError('source definitions missing')
exec(compile(ast.Module(body=_nodes,type_ignores=[]),str(BASE),'exec'),globals())

def tensor_hash(x):
 x=x.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes());return h.hexdigest()
def state_hash(module):
 return hashlib.sha256(json.dumps({k:tensor_hash(v) for k,v in module.state_dict().items()},sort_keys=True).encode()).hexdigest()
def save_state(path,module):np.savez_compressed(path,**{k:v.detach().cpu().numpy() for k,v in module.state_dict().items()})
def freeze(module):
 module.eval()
 for p in module.parameters():p.requires_grad_(False);p.grad=None
 return module
@contextlib.contextmanager
def sqrt64():
 # Same scoped intervention as pinned profile. Main-thread use only, one process
 # per seed, never concurrent optimizer calls or a threaded global override.
 if threading.current_thread() is not threading.main_thread() or threading.active_count()!=1:raise RuntimeError('sqrt profile requires isolated main thread')
 orig_step=torch.optim.AdamW.step;orig_sqrt=torch.Tensor.sqrt;counts={'float32_sqrt_calls':0}
 def stable_sqrt(t):
  if t.dtype==torch.float32:
   counts['float32_sqrt_calls']+=1;return orig_sqrt(t.to(torch.float64)).to(torch.float32)
  return orig_sqrt(t)
 def step(self,*a,**kw):
  if torch.Tensor.sqrt is not orig_sqrt:raise RuntimeError('concurrent sqrt override')
  torch.Tensor.sqrt=stable_sqrt
  try:return orig_step(self,*a,**kw)
  finally:torch.Tensor.sqrt=orig_sqrt
 torch.optim.AdamW.step=step
 try:yield counts
 finally:torch.optim.AdamW.step=orig_step;torch.Tensor.sqrt=orig_sqrt

def optimizer(params):
 return torch.optim.AdamW(params,lr=.008,weight_decay=.00001,betas=(.9,.999),eps=1e-8,amsgrad=False,maximize=False,foreach=False,capturable=False,differentiable=False,fused=False)
def initialize(seed):
 L=nn.ModuleList([TinyRes(64,8,seed['pair1']),TinyRes(64,8,seed['pair2'])]);J=copy.deepcopy(L);C=TinyRes(64,16,seed['comp'])
 if state_hash(L)!=state_hash(J):raise RuntimeError('clone mismatch')
 if any(a.data_ptr()==b.data_ptr() for a,b in zip(L.parameters(),J.parameters())):raise RuntimeError('clone aliases')
 return L,J,C
def schedule(n,seed,steps=600):return torch.randint(0,n,(steps,128),generator=torch.Generator().manual_seed(seed))
def fit_arm(name,module,a,sched,log_path=None):
 opt=optimizer(module.parameters());sh=tensor_hash(sched);input_hash=[tensor_hash(x) for x in a];losses=[];t0=time.monotonic()
 for i,ix in enumerate(sched):
  opt.zero_grad(set_to_none=True)
  if name=='L':loss=F.mse_loss(module[0](a[0][ix]),a[1][ix])+F.mse_loss(module[1](a[1][ix]),a[2][ix])
  elif name=='J':loss=F.mse_loss(module[1](module[0](a[0][ix])),a[2][ix])
  elif name=='C':loss=F.mse_loss(module(a[0][ix]),a[2][ix])
  else:raise ValueError(name)
  if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
  loss.backward()
  if any(p.grad is None or not torch.isfinite(p.grad).all() for p in module.parameters()):raise RuntimeError('nonfinite or missing gradient')
  opt.step()
  if any(not torch.isfinite(p).all() for p in module.parameters()):raise RuntimeError('nonfinite parameter')
  row={'update':i+1,'loss':float(loss.detach())};losses.append(row)
  if log_path:
   with pathlib.Path(log_path).open('a') as f:f.write(json.dumps(row)+'\n')
 steps=[int(opt.state[p]['step']) for p in module.parameters()]
 if steps!=[len(sched)]*len(list(module.parameters())):raise RuntimeError('update count mismatch')
 if sh!=tensor_hash(sched) or input_hash!=[tensor_hash(x) for x in a]:raise RuntimeError('input/schedule mutated')
 return {'parameter_steps':steps,'examples_per_parameter':len(sched)*128,'elapsed_seconds':time.monotonic()-t0,'process_cumulative_max_rss_kib_after_arm':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'rss_semantics':'process lifetime high-water at arm end; not isolated arm peak','final_loss':losses[-1]['loss'],'loss_component_calls':len(sched)*(2 if name=='L' else 1),'schedule_sha256':sh}
def partition_arrays(X,y):
 # Full raw dataset and labels are loaded for original stratification. Only the
 # train/validation feature or label subsets are materialized as tensors.
 ids=np.arange(len(y));train_ids,_=train_test_split(ids,test_size=.25,random_state=1234,stratify=y)
 train_ids,val_ids=train_test_split(train_ids,test_size=.2,random_state=5678,stratify=y[train_ids])
 if len(train_ids)!=1077 or len(val_ids)!=270 or set(train_ids)&set(val_ids):raise RuntimeError('split identity/size')
 return {'Xt':torch.tensor(X[train_ids].astype(np.float32)/16),'yt':torch.tensor(y[train_ids],dtype=torch.long),'Xv':torch.tensor(X[val_ids].astype(np.float32)/16),'yv':torch.tensor(y[val_ids],dtype=torch.long),'train_ids':train_ids,'validation_ids':val_ids}
def load_partition():
 from sklearn.datasets import load_digits
 X,y=load_digits(return_X_y=True);return partition_arrays(X,y)
def feature_metrics(pred,target):
 diff=pred-target;center=target-target.mean(0,keepdim=True)
 mse=float((diff**2).mean());denom=float((center**2).mean())+1e-12
 return {'nmse':mse/denom,'mse':mse,'denominator':denom,'sse':float((diff.double()**2).sum()),'centered_ss':float((center.double()**2).sum()),'elements':target.numel()}
def logits_from_endpoint(teacher,h):
 for block in teacher.blocks[2:]:h=block(h)
 return teacher.head(h)
def assert_environment():
 import scipy,sklearn
 observed={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__}
 expected={'python':'3.13.5','torch':'2.10.0+cpu','numpy':'2.3.5','sklearn':'1.8.0','scipy':'1.17.0'}
 if observed!=expected:raise RuntimeError('environment mismatch '+str(observed))
 if torch.backends.cpu.get_cpu_capability()!='AVX2':raise RuntimeError('not effective AVX2')
 return {**observed,'cpu_capability':torch.backends.cpu.get_cpu_capability(),'environment':{k:os.environ[k] for k in ENV},'mkldnn':torch.backends.mkldnn.enabled,'deterministic':torch.are_deterministic_algorithms_enabled(),'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'torch_build_sha256':hashlib.sha256(torch.__config__.show().encode()).hexdigest()}
def manifest_hashes():return contract.manifest_hashes(HERE)
def source_identity():return contract.package_identity(HERE)
def validate_release(release):return contract.validate_release(release,HERE)

def configure_context(workspace,release):
 global REGISTRY,DEADLINE,WORKSPACE,REPLICATION_ID
 REGISTRY,out=contract.context(workspace,release['replication_id'],HERE)
 DEADLINE=contract.deadline_timestamp(release['deadline_utc'])
 WORKSPACE=pathlib.Path(workspace).resolve();REPLICATION_ID=release['replication_id']
 return out

def run_seed(seed,out):
 out.mkdir(exist_ok=False);start=time.monotonic();dump(out/'START.json',{'id':seed['id'],'seed':seed,'utc':now(),'package_sha256':source_identity(),'files':manifest_hashes()})
 def timeout(*_):raise TimeoutError('seed600second cap')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
 try:
  env=assert_environment();verify_sources();data=load_partition();Xt,yt=data['Xt'],data['yt']
  dump(out/'environment.json',env)
  np.savez_compressed(out/'split_ids.npz',train=data['train_ids'],validation=data['validation_ids'])
  dump(out/'data_hashes.json',{k:tensor_hash(data[k]) for k in ('Xt','yt','Xv','yv')})
  initial=Net(seed['teacher']);save_state(out/'teacher_initial.npz',initial);initial_hash=state_hash(initial);del initial
  with sqrt64() as sqrt_counts:
   teacher=train_model(seed['teacher'],Xt,yt,epochs=60);freeze(teacher);teacher_hash=state_hash(teacher);save_state(out/'teacher_final.npz',teacher)
   with torch.no_grad():a=[x.detach() for x in teacher(Xt,return_acts=True)[1][:3]]
   np.savez_compressed(out/'fit_targets.npz',a0=a[0].numpy(),a1=a[1].numpy(),a2=a[2].numpy())
   L,J,C=initialize(seed);initial_lj=state_hash(L);save_state(out/'pair_initial.npz',L);save_state(out/'comp_initial.npz',C)
   sched=schedule(len(Xt),seed['schedule']);np.save(out/'schedule.npy',sched.numpy());schedule_hash=tensor_hash(sched)
   fits={}
   for name,module in [('L',L),('J',J),('C',C)]:
    untouched={n:state_hash(m) for n,m in [('L',L),('J',J),('C',C)] if n!=name}
    fits[name]=fit_arm(name,module,a,sched,out/f'{name}_updates.jsonl');save_state(out/f'{name}_final.npz',module)
    if untouched!={n:state_hash(m) for n,m in [('L',L),('J',J),('C',C)] if n!=name}:raise RuntimeError('cross-arm mutation')
    if state_hash(teacher)!=teacher_hash or any(p.grad is not None for p in teacher.parameters()):raise RuntimeError('teacher mutation/gradient')
  if sqrt_counts['float32_sqrt_calls']==0:raise RuntimeError('sqrt profile unused')
  # Validation only after every student arm completed; never selects a checkpoint.
  with torch.no_grad():
   tl,va=teacher(data['Xv'],return_acts=True);target=va[2]; teacher_correct=int((tl.argmax(1)==data['yv']).sum())
   endpoints={'L':L[1](L[0](va[0])),'J':J[1](J[0](va[0])),'C':C(va[0])};arm_rows={};saved={'target':target.numpy(),'teacher_logits':tl.numpy(),'labels':data['yv'].numpy()}
   for name,h in endpoints.items():
    logits=logits_from_endpoint(teacher,h)
    if not torch.isfinite(h).all() or not torch.isfinite(logits).all():raise RuntimeError('nonfinite evaluation')
    correct=int((logits.argmax(1)==data['yv']).sum());row=feature_metrics(h,target);row.update(correct=correct,n=270,accuracy=correct/270,quality_pass=row['nmse']<=.08 and correct/270>=teacher_correct/270-.02,state_sha256=state_hash({'L':L,'J':J,'C':C}[name]))
    arm_rows[name]=row;saved[name+'_endpoint']=h.numpy();saved[name+'_logits']=logits.numpy()
   np.savez_compressed(out/'validation_arrays.npz',**saved)
  result={'id':seed['id'],'seed':seed,'status':'COMPLETE','teacher_correct':teacher_correct,'teacher_initial_sha256':initial_hash,'teacher_final_sha256':teacher_hash,'pair_initial_sha256':initial_lj,'schedule_sha256':schedule_hash,'fit_records':fits,'arms':arm_rows,'sqrt_counts':sqrt_counts,'elapsed_seconds':time.monotonic()-start,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'finished_utc':now()}
  dump(out/'result.json',result);dump(out/'FILES_SHA256.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()});return result
 except BaseException as e:
  dump(out/'FAILURE.json',{'id':seed['id'],'status':'FAILED','exception':type(e).__name__,'reason':str(e),'utc':now(),'elapsed_seconds':time.monotonic()-start});raise
 finally:signal.alarm(0)

def analyze(rows):
 if len(rows)!=16 or {r.get('id') for r in rows}!=set(range(16)) or any(r.get('status')!='COMPLETE' for r in rows):return {'decision':'INCOMPLETE','completed':sum(r.get('status')=='COMPLETE' for r in rows),'no_exclusion_or_replacement':True}
 rows=sorted(rows,key=lambda r:r['id']);nm=np.array([math.log2(max(r['arms']['J']['nmse'],1e-12)/max(r['arms']['L']['nmse'],1e-12)) for r in rows]);acc=np.array([(r['arms']['J']['correct']-r['arms']['L']['correct'])/270 for r in rows])
 if not np.isfinite(nm).all() or not np.isfinite(acc).all():return {'decision':'INCOMPLETE','reason':'nonfinite'}
 manifest=json.loads((HERE/'SEEDS.json').read_text());idx=np.random.default_rng(manifest['bootstrap_seed']).integers(0,16,(100000,16));ci=np.quantile(nm[idx].mean(1),[.025,.975]);ai=np.quantile(acc[idx].mean(1),[.025,.975])
 fg='PASS' if ci[1]<0 else 'FAIL' if ci[0]>0 else 'UNCERTAIN';ag='PASS' if ai[0]>-.02 else 'FAIL' if ai[1]<-.02 else 'UNCERTAIN'
 return {'decision':'CONDITIONAL_REGIME_SUPPORTED' if fg==ag=='PASS' else 'NOT_JOINTLY_SUPPORTED','feature_gate':fg,'accuracy_gate':ag,'primary_mean_log2_J_over_L':float(nm.mean()),'primary_ci95':ci.tolist(),'accuracy_mean_J_minus_L':float(acc.mean()),'accuracy_ci95':ai.tolist(),'all_rows':rows,'scope':'fixed-seed matched regimen on reused digits; training-regime bundle, not objective-only or fresh-data confirmation','quality_counts':{a:sum(r['arms'][a]['nmse']<=.08 and r['arms'][a]['correct']/270>=r['teacher_correct']/270-.02 for r in rows) for a in ('L','J','C')},'nmse_sign_J_minus_L':{'lower':int((nm<0).sum()),'equal':int((nm==0).sum()),'higher':int((nm>0).sum())},'bootstrap_seed':manifest['bootstrap_seed'],'bootstrap_draws':100000}

def tree_bytes(root):return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
def rss_bytes(pid):
 try:
  for line in pathlib.Path(f'/proc/{pid}/status').read_text().splitlines():
   if line.startswith('VmRSS:'):return int(line.split()[1])*1024
 except FileNotFoundError:return 0
 return 0
def claim_attempt(registry,record):
 registry.mkdir(parents=True,exist_ok=False)
 dump(registry/'claim.json',record)

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
     env=os.environ.copy();env['CANARIA_MATCHED_TOKEN']=token
     cmd=[sys.executable,str(__file__),'seed','--id',str(s['id']),'--workspace',str(WORKSPACE),'--replication-id',REPLICATION_ID]
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
 finished_utc=now()  # One post-analysis capture binds the deadline check and both completion stamps.
 if contract.deadline_timestamp(finished_utc)>end:abort=abort or 'cohort deadline/cap during finalization'
 if abort is not None or len(records)!=16 or any(r['returncode']!=0 for r in records):summary={'decision':'INCOMPLETE','reason':abort or 'incomplete successful exits','completed_results':len(rows),'no_exclusion_or_replacement':True}
 summary.update(attempts=records,not_started=[s['id'] for s in pending],abort_reason=abort,finished_utc=finished_utc);dump(out/'SUMMARY.json',summary);dump(REGISTRY/'FINISHED.json',{'utc':finished_utc,'decision':summary['decision'],'summary_sha256':sha(out/'SUMMARY.json')});dump(out/'FILES_SHA256.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
 return summary

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('mode',choices=('cohort','seed'))
 ap.add_argument('--workspace',type=pathlib.Path,required=True)
 ap.add_argument('--release',type=pathlib.Path)
 ap.add_argument('--replication-id')
 ap.add_argument('--id',type=int)
 a=ap.parse_args()
 if a.mode=='cohort':
  if a.release is None:ap.error('reviewed new-replication release required')
  release=validate_release(json.loads(a.release.read_text()))
  out=configure_context(a.workspace,release)
  summary=cohort(a.release,out);print(json.dumps({k:v for k,v in summary.items() if k!='all_rows'}))
  if summary['decision']=='INCOMPLETE':raise SystemExit(2)
  return
 if a.replication_id is None or a.id is None:ap.error('internal seed identity required')
 registry,_=contract.context(a.workspace,a.replication_id,HERE)
 r=json.loads((registry/'cohort.json').read_text());token=os.environ.pop('CANARIA_MATCHED_TOKEN','')
 if not token or hashlib.sha256(token.encode()).hexdigest()!=r['token_hash'] or time.time()>=r['end_timestamp']:raise RuntimeError('invalid internal seed invocation')
 release=validate_release(r['release'])
 if release['replication_id']!=a.replication_id:raise RuntimeError('replication identity mismatch')
 out=configure_context(a.workspace,release)
 if out!=pathlib.Path(r['out']):raise RuntimeError('internal output mismatch')
 if not (REGISTRY/f'seed_{a.id}.started.json').exists():raise RuntimeError('seed not admitted')
 seed=next(s for s in json.loads((HERE/'SEEDS.json').read_text())['seeds'] if s['id']==a.id)
 run_seed(seed,out/f'seed_{a.id}')
if __name__=='__main__':main()
