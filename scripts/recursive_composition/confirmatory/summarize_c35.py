from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
SEEDS=list(range(2500,2524)); MIN_ELIGIBLE=16; B=100000; BOOT_SEED=20261335; MARGIN=0.02

def ci_mean(x,idx):
    x=np.asarray(x,float); vals=x[idx].mean(axis=1); return [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--indir',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    rows=[json.loads((a.indir/f'seed_{s}.json').read_text()) for s in SEEDS]; elig=[r for r in rows if r['eligible']]
    checks={'minimum_eligible_met':len(elig)>=MIN_ELIGIBLE,'all_test_flags_false':all(not r.get('test_evaluated',True) for r in rows),'eligible_parameter_match_all':all(all(r['pools'][p]['exact_parameter_match'] for p in ['512','1024']) for r in elig),'eligible_recovery_denominator_positive_all':all(r['scaling_contrast']['valid_recovery_denominators'] for r in elig)}
    summary={'stage':'C35','scientific_role':'confirmatory_practical_equivalence_of_scaling_explanations','seeds':SEEDS,'eligible_seeds':[r['seed'] for r in elig],'ineligible_seeds':[r['seed'] for r in rows if not r['eligible']],'eligibility_count':len(elig),'minimum_required_eligible':MIN_ELIGIBLE,'test_evaluated':False,'checks':checks}
    if elig and checks['eligible_recovery_denominator_positive_all']:
        ea=np.asarray([r['scaling_contrast']['mean_abs_matched_recovery_error'] for r in elig]); ef=np.asarray([r['scaling_contrast']['mean_fraction_matched_recovery_error'] for r in elig]); delta=ea-ef
        rng=np.random.default_rng(BOOT_SEED); idx=rng.integers(0,len(elig),size=(B,len(elig))); ci=ci_mean(delta,idx); est=float(delta.mean()); equiv=bool(ci[0]>-MARGIN and ci[1]<MARGIN)
        if ci[1]<0: direction='FAVORS_ABSOLUTE_COUNT_INVARIANCE'
        elif ci[0]>0: direction='FAVORS_FRACTION_INVARIANCE'
        else: direction='DIRECTIONAL_UNRESOLVED'
        summary['primary']={'E_abs_mean':float(ea.mean()),'E_abs_ci95':ci_mean(ea,idx),'E_frac_mean':float(ef.mean()),'E_frac_ci95':ci_mean(ef,idx),'Delta_E_abs_minus_E_frac':est,'Delta_ci95':ci,'equivalence_margin':[-MARGIN,MARGIN],'equivalence_ci_inside_margin':equiv,'directional_secondary':direction,'per_seed_delta':{str(r['seed']):float(d) for r,d in zip(elig,delta)}}
        primary=bool(all(checks.values()) and equiv); summary['status']='CONFIRMATORY_EQUIVALENCE_PASS' if primary else 'CONFIRMATORY_EQUIVALENCE_FAIL'
    else:
        summary['primary']=None; summary['status']='CONFIRMATORY_EQUIVALENCE_FAIL'
    summary['claim_boundary']='Pass supports practical indistinguishability of the two pre-specified scaling heuristics in this SmallViT/digits intervention only; it does not establish either heuristic as the true mechanism.'
    a.out.write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
