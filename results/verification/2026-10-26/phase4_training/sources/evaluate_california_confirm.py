#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
SEEDS=list(range(2900,2908));BOOT=100000;BOOT_SEED=1355817528

def stat(vals,idx):
    a=np.asarray(vals,dtype=np.float64);m=a[idx].mean(1);lo,hi=np.percentile(m,[2.5,97.5]);return {'mean':float(a.mean()),'ci95':[float(lo),float(hi)],'bootstrap_se':float(m.std(ddof=1))}
def gate_gt(s,margin):
    lo,hi=s['ci95'];return 'PASS' if lo>margin else ('FAIL' if hi<=margin else 'UNCERTAIN')
def main(inputs,out):
    rec=[json.loads(Path(x).read_text()) for x in inputs];errors=[];by={}
    for r in rec:
        s=r.get('seed')
        if s in by:errors.append(f'duplicate {s}')
        by[s]=r
    missing=[s for s in SEEDS if s not in by];unexpected=[s for s in by if s not in SEEDS]
    if missing:errors.append(f'missing {missing}')
    if unexpected:errors.append(f'unexpected {unexpected}')
    rows=[]
    for s in SEEDS:
        if s not in by:continue
        r=by[s]
        try:
            assert r['experiment']=='PHASE4_CALIFORNIA_CONFIRMATORY' and r['evidence_class']=='PROSPECTIVE_CONFIRMATORY'
            assert r['data']['X_sha256']=='ef53061b11adc508abf7f3d57bbeb8a90302caedaa3201b846cdd5f65caddfd5'
            assert r['data']['y_sha256']=='4fcec16744c2835e99b638e9ac95cde0619da1673dfdf7e638d31cd83f32fd14'
            assert r['teacher_recipe']['id']=='low_lr_30' and r['teacher_recipe']['epochs']==30
            assert r['selection_rule']=={'span_nmse_lte':0.12,'test_used_for_selection':False,'val_r2_gte_teacher_minus':0.05}
            assert r['budget_grid_h']==[2,4,6,8,12,16,20,24,32]
            assert r['numeric_profile']['effective_cpu_capability']=='AVX2' and r['numeric_profile']['MKL_CBWR']=='COMPATIBLE' and r['numeric_profile']['threads']==1
            assert r['numeric_profile']['mkldnn'] is False and r['numeric_profile']['deterministic'] is True and r['numeric_profile']['sqrt_intervention']['float32_sqrt_calls']>0
            sep=r['selected_sep'];comp=r['selected_comp'];assert sep is not None and comp is not None
            sb=int(sep['budget']);cb=int(comp['budget']);lr=math.log2(cb/sb);ud=float(comp['comp_test_r2'])-float(sep['sep_test_r2'])
            assert abs(float(r['log2_budget_ratio'])-lr)<=1e-12;assert abs(float(r['test_r2_diff_comp_minus_sep'])-ud)<=1e-12
            vals=[r['teacher_test_r2'],lr,ud];assert all(np.isfinite(float(x)) for x in vals)
            rows.append({'seed':s,'teacher_test_r2':float(r['teacher_test_r2']),'teacher_val_r2':float(r['teacher_val_r2']),'sep_budget':sb,'comp_budget':cb,'log2_budget_ratio':lr,'selected_test_r2_diff':ud,
                         'sep_test_r2':float(sep['sep_test_r2']),'comp_test_r2':float(comp['comp_test_r2'])})
        except Exception as e:errors.append(f'seed{s}:{type(e).__name__}:{e}')
    d={'experiment':'PHASE4_CALIFORNIA_CONFIRMATORY','evidence_class':'PROSPECTIVE_CONFIRMATORY','attempted':len(rec),'eligible':len(rows),'missing':missing,'errors':errors,'bootstrap_resamples':BOOT,'bootstrap_seed':BOOT_SEED}
    if errors or len(rows)!=8:d['decision']='STOP_INTEGRITY_OR_CENSORING'
    else:
        rng=np.random.default_rng(BOOT_SEED);idx=rng.integers(0,8,size=(BOOT,8));t=stat([x['teacher_test_r2'] for x in rows],idx);b=stat([x['log2_budget_ratio'] for x in rows],idx);u=stat([x['selected_test_r2_diff'] for x in rows],idx)
        t['margin']=.70;t['status']=gate_gt(t,.70);lower=sum(x['comp_budget']<x['sep_budget'] for x in rows);ties=sum(x['comp_budget']==x['sep_budget'] for x in rows);b['geometric_mean_ratio']=float(2**b['mean']);b['composed_lower_count']=lower;b['ties']=ties;b['status']='PASS' if lower>=7 and b['ci95'][1]<0 else 'FAIL';u['margin']=-.03;u['status']=gate_gt(u,-.03)
        d.update({'rows':rows,'teacher_quality':t,'composition_budget':b,'selected_utility':u});ss=[t['status'],b['status'],u['status']];d['decision']='PHASE4_CONFIRMATORY_PASS' if all(x=='PASS' for x in ss) else ('PHASE4_CONFIRMATORY_FAIL' if 'FAIL' in ss else 'PHASE4_CONFIRMATORY_UNCERTAIN')
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');print(json.dumps({k:d.get(k) for k in ['decision','attempted','eligible','errors']}));return 0 if d['decision']!='STOP_INTEGRITY_OR_CENSORING' else 2
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();raise SystemExit(main(a.inputs,a.out))
