from __future__ import annotations
import argparse,gc,hashlib,json,os,platform,resource,sys,threading,time
from pathlib import Path
import numpy as np, psutil, sklearn, torch
EXPECTED_ENV={'python':'3.13.5','torch':'2.10.0+cpu','numpy':'2.3.5','scikit_learn':'1.8.0','psutil':'7.2.2'}
TENSOR_HASHES={'compiler_block0.pt':'78fd7f52ef6f019f6ede72c73b4928b0482d61e7f7914de532ebc084779fce56','compiler_block1.pt':'d70098af827694a42e7bb31958cf9959ec3aceef0d432941960c15d0cb5091c8'}
LOGICAL_BLOCK_CHUNKS=4096
EXPECTED_CHECKSUM=234473.0256500244
HEADROOM_BYTES=64*1024*1024
RSS_INTERVAL_S=.001
torch.set_num_threads(1)
def current_env(): return {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scikit_learn':sklearn.__version__,'psutil':psutil.__version__,'platform':platform.platform(),'torch_threads':torch.get_num_threads()}
def env_matches():
 c=current_env(); return all(c[k]==v for k,v in EXPECTED_ENV.items()) and c['torch_threads']==1 and sys.platform.startswith('linux')
def tensor_hash(path):
 sd=torch.load(path,map_location='cpu',weights_only=True); h=hashlib.sha256()
 for k in sorted(sd):
  t=sd[k].detach().cpu().contiguous(); h.update(k.encode()+b'\0'); h.update(str(t.dtype).encode()+b'\0'); h.update(str(tuple(t.shape)).encode()+b'\0'); h.update(t.numpy().tobytes(order='C'))
 return h.hexdigest()
def verify_payload(root):
 obs={n:tensor_hash(root/n) for n in TENSOR_HASHES}; return {'ok':obs==TENSOR_HASHES,'expected':TENSOR_HASHES,'observed':obs}
def touch_state(sd): return sum(float(v.sum()) for v in sd.values() if torch.is_tensor(v))
def start_sampler(proc):
 stop=threading.Event(); samples=[]
 def w():
  while not stop.is_set():
   try:samples.append(proc.memory_info().rss)
   except psutil.Error:break
   time.sleep(RSS_INTERVAL_S)
  try:samples.append(proc.memory_info().rss)
  except psutil.Error:pass
 th=threading.Thread(target=w,daemon=True); th.start(); return stop,th,samples
def memory_failure(exc):
 if isinstance(exc,MemoryError): return True
 s=f'{type(exc).__name__}: {exc}'.lower(); return any(k in s for k in ['memory','alloc','bad_alloc','cannot allocate','out of memory','not enough memory'])
def child_run(root,mode):
 env=current_env(); check=verify_payload(root)
 if not env_matches(): return {'status':'ENVIRONMENT_MISMATCH','environment':env,'payload_check':check}
 if not check['ok']: return {'status':'PAYLOAD_TENSOR_HASH_MISMATCH','environment':env,'payload_check':check}
 warm=torch.load(root/'compiler_block0.pt',map_location='cpu',weights_only=True); warm_checksum=touch_state(warm); del warm; gc.collect()
 proc=psutil.Process(os.getpid()); stop,th,samples=start_sampler(proc); time.sleep(.005); bm=proc.memory_info(); br=int(bm.rss); bv=int(bm.vms)
 constrained=mode in ('full_constrained','streaming_constrained'); limit=None; old=resource.getrlimit(resource.RLIMIT_AS)
 if constrained:
  limit=bv+HEADROOM_BYTES; resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
 retained=[]; checksum=0.; completed=0; error=None; status='SUCCESS'; t0=time.perf_counter()
 try:
  for i in range(LOGICAL_BLOCK_CHUNKS):
   sd=torch.load(root/f'compiler_block{i%2}.pt',map_location='cpu',weights_only=True); checksum+=touch_state(sd)
   if mode in ('full_unconstrained_control','full_constrained'): retained.append(sd)
   elif mode=='streaming_constrained':
    del sd
    if (i+1)%128==0: gc.collect()
   else: raise ValueError(mode)
   completed=i+1
 except BaseException as exc:
  error=f'{type(exc).__name__}: {exc}'; status='MEMORY_ALLOCATION_FAILURE' if memory_failure(exc) else 'UNRELATED_FAILURE'
 elapsed=time.perf_counter()-t0; stop.set(); th.join(timeout=2)
 try: fm=proc.memory_info(); peak=max(samples) if samples else fm.rss; fv=int(fm.vms)
 except psutil.Error: peak=max(samples) if samples else br; fv=None
 return {'status':status,'mode':mode,'environment':env,'payload_check':check,'warm_checksum':warm_checksum,'logical_block_chunks_target':LOGICAL_BLOCK_CHUNKS,'completed_chunks':completed,'checksum':checksum,'expected_checksum':EXPECTED_CHECKSUM,'elapsed_seconds':elapsed,'baseline_rss_bytes':br,'peak_rss_bytes':int(peak),'peak_rss_delta_bytes':int(max(0,peak-br)),'baseline_vms_bytes':bv,'final_vms_bytes':fv,'constrained':constrained,'headroom_bytes':HEADROOM_BYTES if constrained else None,'rlimit_as_bytes':limit,'old_rlimit_as':[old[0],old[1]],'error':error}
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--artifact-dir',type=Path,required=True); ap.add_argument('--child-mode',choices=['full_unconstrained_control','full_constrained','streaming_constrained'],required=True); a=ap.parse_args(); print(json.dumps(child_run(a.artifact_dir,a.child_mode)))
