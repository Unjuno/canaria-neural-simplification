#!/usr/bin/env python3
"""Locked chronological known-seed systems extension. No scientific source modification."""
import os
os.environ.update(ATEN_CPU_CAPABILITY='avx2',MKL_CBWR='COMPATIBLE',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',MKL_DYNAMIC='FALSE',OMP_DYNAMIC='FALSE')
import argparse,copy,hashlib,importlib.util,io,json,platform,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
SEED=int(os.environ['CANARIA_EXTENSION_SEED'])
assert SEED in (871201,871202,871203)
OUT=ROOT/'local_evidence/native_cost_extension'/f'seed_{SEED}'
REF=Path('/workspace/shared/canaria_r87r2_training_20261026/local_evidence/run')/f'seed_{SEED}'
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mkldnn.enabled=False;torch.use_deterministic_algorithms(True)
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
r=load('original',ROOT/'scripts/reproduce/core_discovery_digits/run_confirmatory.py')
d=load('diagnostic',ROOT/'scripts/native_cost_pilot/r87_dispatch.py')
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def snapshot():
 paths=list((ROOT/'scripts/native_cost_pilot').glob('*.py'))+list((ROOT/'scripts/native_cost_extension').glob('*.py'))+[ROOT/'scripts/reproduce/core_discovery_digits/run_confirmatory.py',ROOT/'results/native_cost_extension/PROTOCOL.md',REF/'outcome.json',REF/'record.json']
 dump(OUT/'SOURCE_SHA256.json',{str(p):sha(p) for p in paths})
 raw=(ROOT/'scripts/reproduce/core_discovery_digits/run_confirmatory.py').read_bytes()
 assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()=='8759933ed2bb95014c3afc835acafa1743e40ea6'
def environment():
 return dict(python=sys.version,executable=sys.executable,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),affinity=sorted(os.sched_getaffinity(0)),load=os.getloadavg(),capability=torch.backends.cpu.get_cpu_capability(),threads=torch.get_num_threads(),interop=torch.get_num_interop_threads(),mkldnn=torch.backends.mkldnn.enabled,deterministic=torch.are_deterministic_algorithms_enabled(),env={k:os.environ.get(k) for k in ('MKL_CBWR','ATEN_CPU_CAPABILITY','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},torch_build=torch.__config__.show())
class Active(nn.Module):
 def __init__(self,teacher,kind,h):
  super().__init__();self.stem=copy.deepcopy(teacher.stem);self.head=copy.deepcopy(teacher.head)
  self.blocks=nn.ModuleList(copy.deepcopy(list(teacher.blocks)[2:]))
  self.kind=kind
  if kind=='sep':self.r1=r.TinyRes(64,h,0);self.r2=r.TinyRes(64,h,0)
  else:self.comp=r.TinyRes(64,2*h,0)
 def forward(self,x):
  h=r.F.gelu(self.stem(x))
  h=self.comp(h) if self.kind=='comp' else self.r2(self.r1(h))
  for block in self.blocks:h=block(h)
  return self.head(h)
def artifact(name,model,kind,h,scope):
 state=dict(model.state_dict());bio=io.BytesIO();torch.save(state,bio);data=bio.getvalue()
 manifest={'format':'torch.save_plain_state_dict','model':kind,'h':h,'scope':scope,'dtype':'float32','architecture':'digits64_residual','state_shapes':{k:list(v.shape) for k,v in state.items()}}
 raw=json.dumps(manifest,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
 p=OUT/'artifacts';p.mkdir(exist_ok=True);(p/(name+'.pt')).write_bytes(data);(p/(name+'.json')).write_bytes(raw)
 return dict(name=name,state_bytes=len(data),manifest_bytes=len(raw),total_bytes=len(data)+len(raw),tensor_bytes=sum(v.numel()*v.element_size() for v in state.values()),parameters=sum(p.numel() for p in model.parameters()),state_sha256=hashlib.sha256(data).hexdigest(),manifest_sha256=hashlib.sha256(raw).hexdigest())
def fresh(name):
 p=OUT/'artifacts';m=json.loads((p/(name+'.json')).read_text());model=r.Net(0) if m['model']=='teacher' else Active(r.Net(0),m['model'],m['h'])
 model.load_state_dict(torch.load(p/(name+'.pt'),weights_only=True));return model.eval()
def capture():
 OUT.mkdir(parents=True,exist_ok=True);snapshot();dump(OUT/'START.json',dict(epoch=time.time(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),environment=environment()))
 assert torch.backends.cpu.get_cpu_capability()=='AVX2'
 captured={};fits=[];joint=[]
 orig_train,orig_fit,orig_joint=r.train_model,r.fit_map,r.fit_joint
 def train(*a,**kw):
  m=orig_train(*a,**kw);captured['teacher']=m;captured['teacher_hash']=d.model_hash(m);return m
 def fit(module,X,Y,updates,seed):
  rec={'fit_seed':seed,'updates':updates,'initial_state':d.model_hash(module),'input_hash':d.tensor_hash(X),'target_hash':d.tensor_hash(Y)}
  m=orig_fit(module,X,Y,updates,seed);rec['final_state']=d.model_hash(m);fits.append(rec);captured[seed]=m;return m
 def fit_joint(*a,**kw):
  result=orig_joint(*a,**kw);joint.extend(d.model_hash(m) for m in result);return result
 r.train_model=train;r.fit_map=fit;r.fit_joint=fit_joint
 sm=load('sqrtprofile',ROOT/'scripts/native_cost_pilot/r87_sqrt_profile.py');counts,restore=sm.install_scoped_sqrt64(torch)
 try:
  with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON):torch.mm(torch.ones(8,8),torch.ones(8,8))
  outcome=r.run(SEED)
 finally:restore()
 dump(OUT/'outcome.json',outcome);dump(OUT/'capture_hashes.json',dict(teacher=captured['teacher_hash'],fits=fits,joint=joint,sqrt=counts))
 prior=json.loads((REF/'record.json').read_text())['trace'];previous=json.loads((REF/'outcome.json').read_text())
 gates={'outcome_exact':outcome==previous,'teacher_final_exact':captured['teacher_hash']==prior['teacher']['final_state'],'all_fit_records_exact':fits==prior['fits'],'joint_final_exact':joint==prior['joint_final'],'sqrt_active':counts['float32_sqrt_calls']>0}
 dump(OUT/'SCIENCE_GATE.json',gates)
 if not all(gates.values()):raise RuntimeError('SCIENTIFIC_GATE_MISMATCH: deployment measures blocked')
 teacher=captured['teacher'];teacher.eval();metrics=[artifact('teacher',teacher,'teacher',None,'active_whole')];models={};wrappers={}
 for kind in ('sep','comp'):
  h=outcome['selected_'+kind]['h'];a=Active(teacher,kind,h)
  if kind=='sep':
   m1=captured[SEED+420000+h];m2=captured[SEED+440000+h];a.r1.load_state_dict(m1.state_dict());a.r2.load_state_dict(m2.state_dict());wrapper=r.PairReplacedNet(teacher,0,r1=m1,r2=m2);replacement=nn.Sequential(m1,m2)
  else:
   m=captured[SEED+460000+h];a.comp.load_state_dict(m.state_dict());wrapper=r.PairReplacedNet(teacher,0,comp=m);replacement=m
  models[kind]=a.eval();wrappers[kind]=wrapper.eval()
  metrics.extend([artifact(kind,a,kind,h,'active_whole'),artifact(kind+'_raw',wrapper,kind,h,'raw_wrapper'),artifact(kind+'_replacement',replacement,kind,h,'replacement_only')])
 dump(OUT/'artifact_metrics.json',metrics)
 data=r.datasets();audit=[]
 with torch.inference_mode():
  for kind in ('teacher','sep','comp'):
   model=teacher if kind=='teacher' else models[kind];original=model if kind=='teacher' else wrappers[kind];loaded=fresh(kind)
   for split,X,y in list(zip(('train','validation','test'),data[::2],data[1::2]))+ [('timing_batch128',data[4][:128],data[5][:128]),('timing_batch1',data[4][:1],data[5][:1])]:
    original_y=original(X);active_y=model(X);loaded_y=loaded(X)
    audit.append(dict(model=kind,split=split,wrapper_active_equal=torch.equal(original_y,active_y),active_reload_equal=torch.equal(active_y,loaded_y),wrapper_active_max_abs=float((original_y-active_y).abs().max()),reload_max_abs=float((active_y-loaded_y).abs().max()),accuracy=float((active_y.argmax(-1)==y).float().mean())))
 dump(OUT/'EQUALITY_GATE.json',audit)
 if not all(x['wrapper_active_equal'] and x['active_reload_equal'] for x in audit):raise RuntimeError('EXACT_LOGITS_GATE_MISMATCH: timing blocked')
 torch.save({'batch128':data[4][:128].clone(),'batch1':data[4][:1].clone()},OUT/'inputs.pt')
 dump(OUT/'CAPTURE_COMPLETE.json',dict(epoch=time.time(),status='PASS',environment=environment()))
 print('CAPTURE AND EXACT GATES PASS',flush=True)
def probe(index,cpu):
 assert all(json.loads((OUT/'SCIENCE_GATE.json').read_text()).values())
 assert all(x['wrapper_active_equal'] and x['active_reload_equal'] for x in json.loads((OUT/'EQUALITY_GATE.json').read_text()))
 os.sched_setaffinity(0,{cpu});models={k:fresh(k) for k in ('teacher','sep','comp')};inputs=torch.load(OUT/'inputs.pt',weights_only=True);rng=np.random.default_rng(87132026+index)
 result={'index':index,'environment_start':environment(),'batches':{}}
 with torch.inference_mode():
  for batch in ('batch128','batch1'):
   X=inputs[batch]
   for model in models.values():
    for _ in range(100):model(X)
   rounds=[];first=int(rng.integers(0,2))
   for n in range(30):
    order=('sep','comp') if (n+first)%2==0 else ('comp','sep');row={'round':n,'order':list(order)}
    for kind in order:
     t=time.perf_counter_ns()
     for _ in range(100):models[kind](X)
     row[kind]=time.perf_counter_ns()-t
    rounds.append(row)
   teacher=[]
   for n in range(30):
    t=time.perf_counter_ns()
    for _ in range(100):models['teacher'](X)
    teacher.append(time.perf_counter_ns()-t)
   med={k:float(np.median([x[k] for x in rounds]))/100 for k in ('sep','comp')}
   result['batches'][batch]={'rounds':rounds,'teacher_rounds_ns':teacher,'median_ns_per_forward':med,'comp_sep_ratio':med['comp']/med['sep']}
 result['environment_end']=environment();dump(OUT/f'probe_{index}.json',result);print(json.dumps({b:v['comp_sep_ratio'] for b,v in result['batches'].items()}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('capture','probe'));p.add_argument('--index',type=int);p.add_argument('--cpu',type=int);a=p.parse_args()
 capture() if a.mode=='capture' else probe(a.index,a.cpu)
