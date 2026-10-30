"""Deterministic descriptive report, no resampling or exclusions."""
import hashlib,json,statistics,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'local_evidence/native_cost_pilot';DEST=ROOT/'results/native_cost_pilot'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def classify(r):return 'observed >=5% benefit' if r<=.95 else ('observed <5% benefit' if r<1 else 'no observed benefit')
metrics={x['name']:x for x in read(OUT/'artifact_metrics.json')};probes=[read(OUT/f'probe_{i}.json') for i in range(7)]
assert [x['index'] for x in probes]==list(range(7))
assert all(read(OUT/'SCIENCE_GATE.json').values())
assert all(x['wrapper_active_equal'] and x['active_reload_equal'] for x in read(OUT/'EQUALITY_GATE.json'))
for p in probes:
 for b in ('batch128','batch1'):
  rows=p['batches'][b]['rounds'];assert len(rows)==30
  med={k:statistics.median(x[k] for x in rows)/100 for k in ('sep','comp')}
  assert med==p['batches'][b]['median_ns_per_forward'];assert med['comp']/med['sep']==p['batches'][b]['comp_sep_ratio']
for m in metrics.values():
 for suffix,key in (('pt','state'),('json','manifest')):
  p=OUT/'artifacts'/(m['name']+'.'+suffix);assert p.stat().st_size==m[key+'_bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==m[key+'_sha256']
 m['archive_overhead_bytes']=m['state_bytes']-m['tensor_bytes']
ratio=metrics['comp']['total_bytes']/metrics['sep']['total_bytes'];timing={}
for b in ('batch128','batch1'):
 rr=[p['batches'][b]['comp_sep_ratio'] for p in probes];median=statistics.median(rr)
 timing[b]={'all_seven_ratios':rr,'median_ratio':median,'range':[min(rr),max(rr)],'status':classify(median),'range_crosses_0_95':min(rr)<=.95<max(rr),'per_process_median_ns': [p['batches'][b]['median_ns_per_forward'] for p in probes]}
report={'experiment':'R87R2_NATIVE_COST_PILOT','seed':871200,'evidence_class':'EXPLORATORY_SINGLE_KNOWN_SEED','decision':'PILOT_SUPPORT' if ratio<=.95 and timing['batch128']['median_ratio']<=.95 else 'PILOT_NOT_SUPPORTED','science_gate':'PASS','exact_logits_gate':'PASS','all_seven_probes_complete':True,'byte_ratio':ratio,'byte_status':classify(ratio),'timing':timing,'artifact_metrics':metrics,'ratios':{scope:{key:metrics['comp'+suffix][key]/metrics['sep'+suffix][key] for key in ('total_bytes','tensor_bytes','parameters')} for scope,suffix in (('active_whole',''),('raw_wrapper','_raw'),('replacement_only','_replacement'))},'task_outcome_carried_forward':read(OUT/'outcome.json'),'elapsed_seconds':time.time()-read(OUT/'START.json')['epoch'],'limitations':['One already-known seed, not fresh-cohort evidence','Timing descriptive and noisy; no cross-model confidence interval','Teacher context only; primary comparison active componentwise vs active composed','No RAM, VRAM, energy, generalization or noninferiority claim'],'scientific_or_timing_failed_attempts':[],'ancillary_metadata_errors':['Post-run lscpu unavailable CPU sysfs; retained error and /proc/cpuinfo fallback']}
dump(DEST/'SUMMARY.json',report)
lines=['# R87R2 native cost pilot — completed', '', '**Decision: '+report['decision']+'**. Exploratory known seed871200 only; no fresh cohort or generalization claim.','', 'Original full outcome, teacher, all21 fit records and joint hashes match prior run exactly. All15 exact-logits audits pass (3 models × train/validation/test/batch128/batch1), including fresh save/load. No scientific capture/timing harness failures or scientific reruns.','', '## Independent primary gates',f'- Active whole serialized+manifest bytes: sep {metrics["sep"]["total_bytes"]:,}, comp {metrics["comp"]["total_bytes"]:,}; ratio {ratio:.9f}, reduction {(1-ratio)*100:.3f}%. **{classify(ratio)}**.']
for b in ('batch128','batch1'):
 v=timing[b];lines.extend([f'- {b}: median process ratio {v["median_ratio"]:.9f} ({(1-v["median_ratio"])*100:.3f}% reduction); **{v["status"]}**. Range {v["range"]}; crosses0.95={v["range_crosses_0_95"]}.', '  Seven ratios: '+', '.join(f'{x:.9f}' for x in v['all_seven_ratios'])])
lines.extend(['','## Scope accounting','|Scope/model|Parameters|Tensor payload bytes|Archive overhead|Manifest|Total bytes|','|---|---:|---:|---:|---:|---:|'])
for name,m in metrics.items():lines.append(f'|{name}|{m["parameters"]}|{m["tensor_bytes"]}|{m["archive_overhead_bytes"]}|{m["manifest_bytes"]}|{m["total_bytes"]}|')
lines.extend(['','Both raw wrappers retain two unexecuted teacher blocks; both active arms physically remove them. Replacement-only savings must not be described as whole-model savings.','', 'Seven sequential fresh CPU0-affinity processes, float32 eval/inference_mode, AVX2/MKL COMPATIBLE, MKLDNN off, one intra/inter-op thread. Per arm/batch100 warmups,30 alternating paired rounds of100 forwards. Teacher timed separately. No outlier filtering. Ratio-of-per-arm-medians within process, median of7 process ratios.','', 'All raw durations, environments, manifests, state_dict artifacts, source/reference hashes and logs are in `local_evidence/native_cost_pilot/`. The original utility outcomes are carried forward unchanged in SUMMARY.json. Timing is descriptive/noisy; no seed bootstrap, noninferiority, RAM/VRAM/energy claim or across-model statistical inference.'])
(DEST/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:report[k] for k in ('decision','byte_ratio','byte_status','timing')}))
