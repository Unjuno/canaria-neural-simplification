"""Frozen census and bounded chronological sequential supervisor."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'local_evidence/native_cost_extension'; DEST=ROOT/'results/native_cost_extension'; REF=Path('/workspace/shared/canaria_r87r2_training_20261026/local_evidence/run')
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
rows=[]
for seed in range(871200,871216):
 p=REF/f'seed_{seed}/outcome.json';o=json.loads(p.read_text());a={k:38218+o['selected_'+k]['budget'] for k in ('sep','comp')}
 rows.append(dict(seed=seed,reference_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),selected_h={k:o['selected_'+k]['h'] for k in a},replacement_budget={k:o['selected_'+k]['budget'] for k in a},active_whole_parameters=a,float32_tensor_payload_bytes={k:4*v for k,v in a.items()},comp_sep_payload_ratio=a['comp']/a['sep'],payload_gate_5percent=a['comp']/a['sep']<=.95))
dump(DEST/'CENSUS.json',{'evidence_class':'ANALYTIC_FROZEN_ENDPOINT_PAYLOAD_NOT_ARCHIVE_OR_LATENCY','fixed_shell_parameters':38218,'rows':rows})
start=time.time();deadline=start+45*60;cpu=min(os.sched_getaffinity(0));attempts=[]
dump(OUT/'START.json',dict(epoch=start,deadline_epoch=deadline,cpu=cpu,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),science_concurrency='Parent confirmed no other science'))
for seed in (871201,871202,871203):
 dest=OUT/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True);env=dict(os.environ,CANARIA_EXTENSION_SEED=str(seed))
 for mode,index in [('capture',None)]+[('probe',i) for i in range(7)]:
  remaining=deadline-time.time()-60
  if remaining<=0:
   dump(OUT/'INCOMPLETE_EXECUTION.json',dict(reason='45min cap',seed=seed,mode=mode,index=index));sys.exit(2)
  cmd=[sys.executable,str(Path(__file__).with_name('pilot.py')),mode]
  if index is not None:cmd+=['--index',str(index),'--cpu',str(cpu)]
  rec=dict(seed=seed,mode=mode,index=index,argv=cmd,start_epoch=time.time())
  with (dest/(mode+('' if index is None else '_'+str(index))+'_attempt_1.log')).open('w') as log:
   try:rec['returncode']=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(900,remaining)).returncode
   except subprocess.TimeoutExpired:rec['failure']='TIMEOUT_INCOMPLETE_EXECUTION'
  rec['end_epoch']=time.time();attempts.append(rec);dump(OUT/'EXECUTION.json',attempts);print(json.dumps(rec),flush=True)
  if rec.get('returncode')!=0:break
 else:
  # Post-timing custody only, no additional scientific training or timing.
  cmd=[sys.executable,str(Path(__file__).with_name('custody.py'))]
  with (dest/'custody.log').open('w') as log:rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  dump(dest/'CUSTODY_EXIT.json',dict(returncode=rc))
dump(OUT/'COMPLETE.json',dict(epoch=time.time(),elapsed_seconds=time.time()-start,all_science_attempts_succeeded=all(r.get('returncode')==0 for r in attempts),attempt_count=len(attempts)))
