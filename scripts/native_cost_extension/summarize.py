import hashlib,json,statistics,time
from pathlib import Path
R=Path('/workspace/shared/canaria_native_cost_extension_20261026');O=R/'local_evidence/native_cost_extension';D=R/'results/native_cost_extension'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def cls(x):return 'observed >=5% benefit' if x<=.95 else 'observed <5% benefit' if x<1 else 'no observed benefit'
census=read(D/'CENSUS.json'); ratios=[x['comp_sep_payload_ratio'] for x in census['rows']]
census['summary']={'arithmetic_mean_ratio':statistics.mean(ratios),'geometric_mean_ratio':statistics.geometric_mean(ratios),'range':[min(ratios),max(ratios)],'count_meeting_5percent_payload_gate':sum(x<=.95 for x in ratios)}
for row in census['rows']:row['replacement_only_ratio']=row['replacement_budget']['comp']/row['replacement_budget']['sep']
census['scope_audit']={'shell_components':{'stem':4160,'remaining_blocks':2*16704,'head':650},'source_buffer_registration':False,'dtype_bytes':4,'archive_bytes_for_unmaterialized_seeds':'NOT MEASURED','latency_for_unmaterialized_seeds':'NOT MEASURED'}
dump(D/'CENSUS.json',census)
reports=[]
for seed in (871201,871202,871203):
 p=O/f'seed_{seed}';sg=read(p/'SCIENCE_GATE.json');eg=read(p/'EQUALITY_GATE.json');assert all(sg.values()) and all(x['wrapper_active_equal'] and x['active_reload_equal'] for x in eg)
 metrics={m['name']:m for m in read(p/'artifact_metrics.json')}
 for n,m in metrics.items():
  for suffix,key in [('pt','state'),('json','manifest')]:
   f=p/'artifacts'/f'{n}.{suffix}';assert f.stat().st_size==m[key+'_bytes'];assert hashlib.sha256(f.read_bytes()).hexdigest()==m[key+'_sha256']
  m['archive_overhead_bytes']=m['state_bytes']-m['tensor_bytes'];m['total_nonpayload_bytes']=m['total_bytes']-m['tensor_bytes']
  assert m['tensor_bytes']==4*m['parameters']
 row=next(x for x in census['rows'] if x['seed']==seed)
 for k in ('sep','comp'):assert row['active_whole_parameters'][k]==metrics[k]['parameters']
 probes=[read(p/f'probe_{i}.json') for i in range(7)];timing={}
 for b in ('batch128','batch1'):
  ratios=[]
  for i,q in enumerate(probes):
   assert q['index']==i;rs=q['batches'][b]['rounds'];assert len(rs)==30
   med={k:statistics.median(x[k] for x in rs)/100 for k in ('sep','comp')};assert med==q['batches'][b]['median_ns_per_forward'];r=med['comp']/med['sep'];assert r==q['batches'][b]['comp_sep_ratio'];ratios.append(r)
  median=statistics.median(ratios);timing[b]={'all_seven_ratios':ratios,'median_ratio':median,'range':[min(ratios),max(ratios)],'range_crosses_0_95':min(ratios)<=.95<max(ratios),'status':cls(median)}
 ratio=metrics['comp']['total_bytes']/metrics['sep']['total_bytes'];delta=metrics['sep']['total_bytes']-metrics['comp']['total_bytes'];pay=metrics['sep']['tensor_bytes']-metrics['comp']['tensor_bytes'];over=metrics['sep']['total_nonpayload_bytes']-metrics['comp']['total_nonpayload_bytes'];assert delta==pay+over
 reports.append({'seed':seed,'science_gate':'PASS','exact_logits_gate':'PASS','byte_ratio':ratio,'byte_status':cls(ratio),'joint_5percent_support':ratio<=.95 and timing['batch128']['median_ratio']<=.95,'timing':timing,'artifact_metrics':metrics,'overhead_reconciliation':{'total_bytes_saved':delta,'tensor_payload_saved':pay,'nonpayload_saved':over},'task_outcome_carried_forward':read(p/'outcome.json')})
report={'evidence_class':'EXPLORATORY_TECHNICAL_KNOWN_SEED_EXTENSION_NOT_CONFIRMATORY_OR_NEW_N','seeds':reports,'execution':read(O/'COMPLETE.json'),'census_summary':census['summary'],'limitations':['Three chronological already-known seeds, not fresh cohort','No pooling with pilot to claim universality','Technical repeats are not independent scientific seeds','No inference of runtime or saved-file size from parameter payload','No RAM, VRAM, energy, noninferiority or generalization claim']}
dump(D/'SUMMARY.json',report)
lines=['# R87R2 native cost exploratory known-seed extension','', '**Not confirmatory; no new scientific n.** All three chronological seeds retained. Prior pilot871200 remains independently not supported; no pooled universality claim.','', '## All16 frozen endpoint analytic payload census',f"Mean whole-payload ratio {census['summary']['arithmetic_mean_ratio']:.9f}; geometric mean {census['summary']['geometric_mean_ratio']:.9f}; range {census['summary']['range']}; 2/16 meet5% payload reduction. These are analytic float32 tensor bytes, not archive size or runtime.",'','|Seed|Sep B|Comp B|Sep whole parameters|Comp whole parameters|Sep payload bytes|Comp payload bytes|Ratio|','|---|---:|---:|---:|---:|---:|---:|---:|']
for x in census['rows']:lines.append(f"|{x['seed']}|{x['replacement_budget']['sep']}|{x['replacement_budget']['comp']}|{x['active_whole_parameters']['sep']}|{x['active_whole_parameters']['comp']}|{x['float32_tensor_payload_bytes']['sep']}|{x['float32_tensor_payload_bytes']['comp']}|{x['comp_sep_payload_ratio']:.9f}|")
lines+=['','## Three chronological measured seeds','|Seed|Serialized+manifest sep/comp bytes|Byte ratio|Batch128 median ratio|Batch1 median ratio|Both5% gates|','|---|---:|---:|---:|---:|---|']
for x in reports:
 m=x['artifact_metrics'];lines.append(f"|{x['seed']}|{m['sep']['total_bytes']}/{m['comp']['total_bytes']}|{x['byte_ratio']:.9f}|{x['timing']['batch128']['median_ratio']:.9f}|{x['timing']['batch1']['median_ratio']:.9f}|{x['joint_5percent_support']}|")
for x in reports:
 lines+=['',f"### Seed{x['seed']}",'Full original outcome/teacher/all fit/joint hashes exact; all15 logits audits exact.']
 for b,t in x['timing'].items():lines.append(f"- {b}: {t['status']}; seven ratios {t['all_seven_ratios']}; range {t['range']}; crosses0.95={t['range_crosses_0_95']}.")
 lines.append('- Byte decomposition: '+str(x['overhead_reconciliation']))
lines+=['','All raw durations, binaries, source/reference/dataset hashes and logs retained locally. Seven sequential fresh single-thread CPU processes per seed; identical pilot serialization and timing estimator. No retries, exclusions, threshold changes or scientific source edits. Task utility is unchanged and carried forward in SUMMARY.json.']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(report,indent=2))
