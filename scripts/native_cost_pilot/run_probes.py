"""Sequential process orchestration with retained failures and total execution cap."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'local_evidence/native_cost_pilot'
start=json.loads((OUT/'START.json').read_text())['epoch'];deadline=start+25*60
cpu=min(os.sched_getaffinity(0));attempts=[]
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
dump(OUT/'TIMING_SUPERVISOR.json',{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cpu':cpu,'deadline_epoch':deadline,'start_epoch':time.time(),'initial_load':os.getloadavg(),'science_concurrency':'Parent confirmed no other science; sequential child probes'})
for i in range(7):
 remaining=deadline-time.time()-60
 if remaining<=0:raise RuntimeError('INCOMPLETE_EXECUTION: total cap reporting reserve')
 cmd=[sys.executable,str(Path(__file__).with_name('pilot.py')),'probe','--index',str(i),'--cpu',str(cpu)]
 began=time.time()
 with (OUT/f'probe_{i}_attempt_1.log').open('w') as log:
  try:
   run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=min(900,remaining));record={'index':i,'returncode':run.returncode,'argv':cmd,'start_epoch':began,'end_epoch':time.time()}
  except subprocess.TimeoutExpired:
   record={'index':i,'failure':'TIMEOUT_INCOMPLETE_EXECUTION','argv':cmd,'start_epoch':began,'end_epoch':time.time()}
 attempts.append(record);dump(OUT/'PROBE_EXECUTION.json',attempts)
 if record.get('returncode')!=0:raise RuntimeError('Retained execution failure; no silent retry')
 print('PROBE_COMPLETE',i,flush=True)
