#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np

SEEDS=list(range(2400,2408));BOOT=100000;BOOT_SEED=2225163077

def finite(x):return np.isfinite(float(x))
def ci(arr,idx):
    vals=np.asarray(arr,dtype=np.float64);means=vals[idx].mean(axis=1);lo,hi=np.percentile(means,[2.5,97.5]);return {'mean':float(vals.mean()),'ci95':[float(lo),float(hi)]}
def gate_gt(stat,margin):
    lo,hi=stat['ci95']
    return 'PASS' if lo>margin else ('FAIL' if hi<=margin else 'UNCERTAIN')
def main(paths,out):
    records=[json.loads(Path(p).read_text()) for p in paths]
    errors=[];by={}
    for r in records:
        s=r.get('seed')
        if s in by:errors.append(f'duplicate seed {s}')
        by[s]=r
    missing=[s for s in SEEDS if s not in by];unexpected=[s for s in by if s not in SEEDS]
    if missing:errors.append(f'missing {missing}')
    if unexpected:errors.append(f'unexpected {unexpected}')
    rows=[]
    for s in SEEDS:
        if s not in by:continue
        r=by[s]
        try:
            assert r['experiment']=='PHASE3B_STRONGER_TEACHER_CONFIRMATORY'
            assert r['evidence_class']=='PROSPECTIVE_CONFIRMATORY'
            assert r['teacher_epochs']==25 and r['map_updates']==600
            assert r['budget_grid_h']==[2,4,6,8,12,16,20,24,32]
            assert r['selection_rule']['span_nmse_lte']==0.12 and r['selection_rule']['val_r2_gte_teacher_minus']==0.05
            assert r['selection_rule']['test_used_for_selection'] is False
            assert r['numeric_profile']['effective_cpu_capability']=='AVX2'
            assert r['numeric_profile']['MKL_CBWR']=='COMPATIBLE'
            assert r['numeric_profile']['torch_threads']==1
            assert r['numeric_profile']['mkldnn_enabled'] is False
            assert r['numeric_profile']['deterministic_algorithms'] is True
            assert r['numeric_profile']['sqrt_intervention']['float32_sqrt_calls']>0
            sep=r['selected_sep'];comp=r['selected_comp']
            assert sep is not None and comp is not None
            sb=int(sep['budget']);cb=int(comp['budget'])
            log2=math.log2(cb/sb)
            assert abs(float(r['log2_budget_ratio'])-log2)<=1e-12
            td=float(comp['comp_test_r2'])-float(sep['sep_test_r2'])
            assert abs(float(r['test_r2_diff_comp_minus_sep'])-td)<=1e-12
            vals=[r['candidate_teacher_test_r2'],r['baseline60_teacher_test_r2'],r['candidate_minus_baseline_test_r2'],log2,td]
            assert all(finite(x) for x in vals)
            rows.append({'seed':s,'candidate_test_r2':float(r['candidate_teacher_test_r2']),'baseline_test_r2':float(r['baseline60_teacher_test_r2']),
                         'teacher_gain':float(r['candidate_minus_baseline_test_r2']),'sep_budget':sb,'comp_budget':cb,'log2_budget_ratio':log2,
                         'selected_test_r2_diff':td})
        except Exception as e:errors.append(f'seed {s}: {type(e).__name__}: {e}')
    decision={'experiment':'PHASE3B_STRONGER_TEACHER_CONFIRMATORY','evidence_class':'PROSPECTIVE_CONFIRMATORY','attempted':len(records),'eligible':len(rows),'missing':missing,'errors':errors,
              'bootstrap_resamples':BOOT,'bootstrap_seed':BOOT_SEED}
    if errors or len(rows)!=8:
        decision['decision']='STOP_INTEGRITY_OR_CENSORING'
    else:
        rng=np.random.default_rng(BOOT_SEED);idx=rng.integers(0,8,size=(BOOT,8))
        candidate=ci([x['candidate_test_r2'] for x in rows],idx)
        gain=ci([x['teacher_gain'] for x in rows],idx)
        log2=ci([x['log2_budget_ratio'] for x in rows],idx)
        utility=ci([x['selected_test_r2_diff'] for x in rows],idx)
        candidate['margin']=0.20;candidate['status']=gate_gt(candidate,0.20)
        gain['margin']=0.05;gain['status']=gate_gt(gain,0.05)
        lower=sum(x['comp_budget']<x['sep_budget'] for x in rows);ties=sum(x['comp_budget']==x['sep_budget'] for x in rows)
        budget_status='PASS' if lower>=7 and log2['ci95'][1]<0 else 'FAIL'
        log2['status']=budget_status;log2['composed_lower_count']=lower;log2['ties']=ties;log2['geometric_mean_ratio']=float(2.0**log2['mean'])
        utility['margin']=-0.05;utility['status']=gate_gt(utility,-0.05)
        decision.update({'rows':rows,'teacher_absolute':candidate,'teacher_improvement':gain,'composition_budget':log2,'selected_utility':utility})
        statuses=[candidate['status'],gain['status'],budget_status,utility['status']]
        decision['decision']='PHASE3B_CONFIRMATORY_PASS' if all(x=='PASS' for x in statuses) else ('PHASE3B_CONFIRMATORY_FAIL' if 'FAIL' in statuses else 'PHASE3B_CONFIRMATORY_UNCERTAIN')
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:decision.get(k) for k in ('decision','attempted','eligible','errors')}),flush=True)
    return 0 if decision['decision']!='STOP_INTEGRITY_OR_CENSORING' else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();raise SystemExit(main(a.inputs,a.out))
