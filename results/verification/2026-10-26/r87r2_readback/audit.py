#!/usr/bin/env python3
import json, math, sys, platform, hashlib, importlib.util, datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'source/results/reproduction/r87r2_fresh_sqrt_profile'
read=lambda p:json.loads(p.read_text())
protocol=read(BASE/'PROTOCOL.json'); container=read(BASE/'FRESH_ROWS.json'); original=read(BASE/'DECISION.json')
rows=sorted(container['rows'],key=lambda r:r['seed'])
assert len(rows)==16 and [r['seed'] for r in rows]==protocol['fresh_seeds']==list(range(871200,871216))
assert container['primary_host']=='a'
errors=[]; selections=[]; rebuilt=[]
def check(ok,context):
 if not ok:errors.append(context)
for row in rows:
 seed=row['seed'];grid=row['grid']; check([g['budget'] for g in grid]==[512,1024,1536,2048,3072,4096,6144],f'{seed}:budget grid')
 check(all(math.isfinite(x) for g in grid for x in g.values() if isinstance(x,(float,int))),f'{seed}:nonfinite')
 check(all(math.isfinite(row[k]) and 0<=row[k]<=1 for k in ['teacher_val_acc','teacher_test_acc']),f'{seed}:teacher finite/range')
 selected={}
 for mode in ['sep','comp']:
  passes=[g[f'{mode}_nmse']<=.08 and g[f'{mode}_val_acc']>=row['teacher_val_acc']-.02 for g in grid]
  for g,p in zip(grid,passes):check(g[f'{mode}_pass']==p,f'{seed}:{mode}:{g["budget"]}:pass')
  picked=next((g for g,p in zip(grid,passes) if p),None);check(picked is not None,f'{seed}:{mode}:unselected')
  # Original runner snapshots sep before adding comp_test_acc when both first pass together.
  expected=dict(picked)
  if mode=='sep':expected.pop('comp_test_acc',None)
  check(row[f'selected_{mode}']==expected,f'{seed}:{mode}:selected object snapshot')
  check([g['budget'] for g in grid if f'{mode}_test_acc' in g]==[picked['budget']],f'{seed}:{mode}:test placement')
  check(math.isfinite(picked[f'{mode}_test_acc']) and 0<=picked[f'{mode}_test_acc']<=1,f'{seed}:{mode}:test finite/range')
  selected[mode]=picked
 log=math.log2(selected['comp']['budget']/selected['sep']['budget']);delta=selected['comp']['comp_test_acc']-selected['sep']['sep_test_acc']
 check(log==row['log2_budget_ratio'],f'{seed}:log ratio');check(delta==row['test_acc_diff_comp_minus_sep'],f'{seed}:test delta')
 rebuilt.append(dict(seed=seed,selected_sep=selected['sep'],selected_comp=selected['comp'],teacher_val_acc=row['teacher_val_acc'],teacher_test_acc=row['teacher_test_acc']))
 selections.append(dict(seed=seed,componentwise_budget=selected['sep']['budget'],composed_budget=selected['comp']['budget'],log2_ratio=log,test_delta=delta))
assert not errors,errors
# Separate implementation from protocol; shared index matrix and fixed NumPy PCG64 seed.
sep=np.asarray([r['componentwise_budget'] for r in selections],dtype=float);comp=np.asarray([r['composed_budget'] for r in selections],dtype=float)
logs=np.log2(comp/sep);delta=np.asarray([r['test_delta'] for r in selections]);n=len(rows)
indices=np.random.Generator(np.random.PCG64(protocol['T']['bootstrap_seed'])).integers(0,n,(protocol['T']['bootstrap_resamples'],n))
def stats(values):
 draws=np.average(values[indices],axis=1);bounds=np.quantile(draws,[.025,.975],method='linear')
 return dict(mean=float(np.average(values)),median=float(np.median(values)),ci95=bounds.tolist(),bootstrap_se=float(np.std(draws,ddof=1)))
p=stats(logs);s=stats(delta)
p['status']='PASS' if p['ci95'][1]<0 else 'FAIL' if p['ci95'][0]>0 else 'UNCERTAIN'
s['status']='PASS' if s['ci95'][0]>-.02 else 'FAIL' if s['ci95'][1]<-.02 else 'UNCERTAIN'
status='PASS' if p['status']==s['status']=='PASS' else 'FAIL' if 'FAIL' in [p['status'],s['status']] else 'UNCERTAIN'
decision=dict(experiment=protocol['experiment'],evidence_class='PROSPECTIVE_CONFIRMATORY',decision='R87R2_CONFIRMATORY_'+status,attempted=n,eligible=n,primary_log2_budget_ratio=p,secondary_test_accuracy_difference=s,mean_componentwise_budget=float(np.average(sep)),mean_composed_budget=float(np.average(comp)),geometric_budget_ratio=float(2**np.average(logs)),geometric_budget_ratio_ci95=[float(2**v) for v in p['ci95']],composed_lower_count=int(sum(comp<sep)),teacher_val_mean=float(np.average([r['teacher_val_acc'] for r in rows])),teacher_test_mean=float(np.average([r['teacher_test_acc'] for r in rows])),bootstrap_resamples=protocol['T']['bootstrap_resamples'],bootstrap_seed=protocol['T']['bootstrap_seed'],test_evaluated=True,test_used_for_selection=False,independent_dataset=False,historical_archive_recovered=False)
def diffs(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  out=[]
  for k in sorted(a.keys()|b.keys()):out+=diffs(a.get(k),b.get(k),path+'/'+k)
  return out
 if isinstance(a,list) and isinstance(b,list):
  if len(a)!=len(b):return [dict(path=path,a=a,b=b)]
  return [d for i,(x,y) in enumerate(zip(a,b)) for d in diffs(x,y,path+'/'+str(i))]
 return [] if a==b else [dict(path=path,a=a,b=b)]
# Saved evaluator only imports standard library at module scope; do not run its execution branches.
f=ROOT/'source/scripts/reproduction_diagnostics/r87r2_confirm.py';spec=importlib.util.spec_from_file_location('saved_evaluator',f);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert 'torch' not in sys.modules
saved=module.evaluate(rebuilt);module.selftest();assert 'torch' not in sys.modules
result=dict(audit='R87R2_SAVED_CANDIDATE_TO_DECISION',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS' if not diffs(original,decision) and not diffs(original,saved) else 'MISMATCH',candidate_count=sum(len(r['grid']) for r in rows),validation_booleans_checked=2*sum(len(r['grid']) for r in rows),first_passing_endpoints_checked=2*n,selection_errors=errors,archive_vs_independent_differences=diffs(original,decision),archive_vs_saved_evaluator_differences=diffs(original,saved),saved_evaluator_synthetic_selftest='PASS',torch_loaded=False,environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),limitations=['Saved scalar metrics only; no raw predictions or model weights audited.','Chronological preregistration and absence of test-informed source/profile selection not established.','Test-field placement is consistent with selection code but cannot prove runtime test-access history.','Protocol/evaluator provenance/evidence-class fields and test_used_for_selection=false are declarations, not independently verified historical facts.','Launch gate and tested-host trace/state-hash parity not audited here.','Seed bootstrap conditional on reused digits split; no independent dataset or task-general inference.','Fresh numeric recipe and seed cohort are separate from historical 1200–1207 archive.'])
for name,obj in [('INDEPENDENT_DECISION.json',decision),('SAVED_EVALUATOR_DECISION.json',saved),('SELECTIONS.json',selections),('AUDIT_RESULT.json',result)]:
 (ROOT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps(result,indent=2));print(json.dumps(decision,indent=2))
assert result['status']=='PASS'
