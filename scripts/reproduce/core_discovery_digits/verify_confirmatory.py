"""Strict reproduction of known seeds; never relabel a numerical mismatch as PASS."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / 'scripts/reproduce/core_discovery_digits/run_confirmatory.py'
SUMMARY = ROOT / 'results/core_discovery_digits/confirm_summary.json'
PROTOCOL = ROOT / 'results/core_discovery_digits/PROTOCOL_LOCK.json'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, tol: float) -> bool:
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b)) <= tol


def bootstrap_ci(values, reps: int, seed: int) -> list[float]:
    a = np.asarray(values, dtype=float)
    if len(a) == 0 or not np.isfinite(a).all():
        raise ValueError('empty or nonfinite bootstrap input')
    ix = np.random.default_rng(seed).integers(0, len(a), (reps, len(a)))
    return list(map(float, np.percentile(a[ix].mean(axis=1), [2.5, 97.5])))


def compare_rows(rows: list[dict], expected: dict, protocol: dict) -> tuple[list[str], dict]:
    """Same numerical acceptance tolerances as the original verifier, plus integrity checks."""
    errors: list[str] = []
    seeds = protocol['confirmatory_seeds']
    got_seeds = [r['seed'] for r in rows]
    if len(seeds) != len(set(seeds)) or len(got_seeds) != len(set(got_seeds)):
        return ['duplicate seed'], {}
    if sorted(got_seeds) != sorted(seeds):
        return ['missing or unexpected seed; complete original cohort required'], {}
    exp = {r['seed']: r for r in expected['per_seed_selected_budgets']}
    if set(exp) != set(seeds):
        return ['summary/protocol cohort mismatch'], {}
    for r in rows:
        e = exp[r['seed']]
        for key in ('componentwise', 'composed'):
            if type(r[key]) is not int or r[key] != e[key]:
                errors.append(f"seed {r['seed']}: {key} {r[key]} != {e[key]}")
        for key, tol in [('log2_ratio', 1e-12), ('test_acc_diff', 1e-6)]:
            if not close(r[key], e[key], tol):
                errors.append(f"seed {r['seed']}: {key} {r[key]} != {e[key]} within {tol}")
    numeric = np.array([[r[k] for k in ('componentwise','composed','log2_ratio','test_acc_diff')] for r in rows],dtype=float)
    if not np.isfinite(numeric).all():
        return errors+['nonfinite selected-row values'], {}
    sep,comp,ratios,test = numeric.T
    reps, bs = protocol['primary']['bootstrap_reps'], protocol['primary']['bootstrap_seed']
    ci, tci = bootstrap_ci(ratios,reps,bs), bootstrap_ci(test,reps,bs)
    agg={'mean_componentwise_budget':float(sep.mean()),'mean_composed_budget':float(comp.mean()),
         'mean_log2_budget_ratio':float(ratios.mean()),'geometric_mean_budget_ratio':float(2**ratios.mean()),
         'bootstrap95_mean_log2_budget_ratio':ci,'mean_test_acc_diff':float(test.mean()),
         'bootstrap95_mean_test_acc_diff':tci,'composed_lower_count':int((comp<sep).sum()),
         'directional_primary_gate':bool(ci[1]<0),'directional_secondary_gate':bool(tci[0]>-0.02)}
    p,s=expected['primary'],expected['secondary_confirmatory']
    checks=[('mean_componentwise_budget',p['mean_componentwise_budget'],0),('mean_composed_budget',p['mean_composed_budget'],0),
            ('mean_log2_budget_ratio',p['mean'],1e-12),('geometric_mean_budget_ratio',p['geometric_mean_budget_ratio'],1e-12),
            ('mean_test_acc_diff',s['mean'],1e-6)]
    for k,w,tol in checks:
        if not close(agg[k],w,tol):errors.append(f'aggregate {k}: {agg[k]} != {w} within {tol}')
    for i in range(2):
        if not close(ci[i],p['bootstrap95'][i],1e-12):errors.append(f'primary bootstrap endpoint {i} mismatch')
    if agg['composed_lower_count'] != p['wins_composed_lower_budget']:errors.append('composed-lower count mismatch')
    return errors,agg


def selected_row(result: dict, seed: int, protocol: dict) -> dict:
    if type(result.get('seed')) is not int or result['seed'] != seed:
        raise ValueError('seed identifier mismatch')
    grid = result['grid']
    if [x['budget'] for x in grid] != protocol['replacement_grammar']['budget_grid']:
        raise ValueError('candidate budget grid mismatch')
    for key in ('sep','comp'):
        selected=result['selected_'+key]
        passing=[g for g in grid if g[key+'_pass']]
        if not passing or selected is None or selected['budget'] != passing[0]['budget']:
            raise ValueError('selected budget is not first passing grid point')
        for g in grid:
            ok=g[key+'_nmse']<=0.08 and g[key+'_val_acc']>=result['teacher_val_acc']-0.02
            if type(g[key+'_pass']) is not bool or g[key+'_pass'] != ok:
                raise ValueError('candidate passing flag inconsistent with locked validation rule')
    sep,comp=result['selected_sep'],result['selected_comp']
    ratio=math.log2(comp['budget']/sep['budget'])
    diff=comp['comp_test_acc']-sep['sep_test_acc']
    if not close(ratio,result['log2_budget_ratio'],1e-12) or not close(diff,result['test_acc_diff_comp_minus_sep'],1e-12):
        raise ValueError('selected endpoint arithmetic inconsistent')
    return {'seed':seed,'componentwise':sep['budget'],'composed':comp['budget'],'log2_ratio':ratio,'test_acc_diff':diff}


def environment() -> dict:
    d={'python':platform.python_version(),'platform':platform.platform()}
    for n in ('numpy','torch','scikit-learn'):
        d[n]=importlib.metadata.version(n)
    return d


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--raw-dir',type=Path,help='New empty output directory; all outputs/logs retained')
    group.add_argument('--input-dir',type=Path,help='Read existing raw seed files; does not execute training')
    args=ap.parse_args()
    expected=json.loads(SUMMARY.read_text());protocol=json.loads(PROTOCOL.read_text())
    seeds=protocol['confirmatory_seeds']
    inputs={str(p.relative_to(ROOT)):digest(p) for p in (RUNNER,SUMMARY,PROTOCOL)}
    started=datetime.now(timezone.utc).isoformat()
    raw=args.input_dir or args.raw_dir or (args.out.parent/(args.out.stem+'-raw'))
    if args.input_dir is None:
        if raw.exists() and any(raw.iterdir()):
            ap.error('raw directory is nonempty; refuse to overwrite. Use --input-dir to recheck stored outputs.')
        raw.mkdir(parents=True,exist_ok=True)
        env=environment()
        import torch
        env['torch_build']=torch.__config__.show();env['cuda_build']=torch.version.cuda
        (raw/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    else:
        if not raw.is_dir():ap.error('input directory missing')
        envpath=raw/'environment.json'
        env=json.loads(envpath.read_text()) if envpath.exists() else {'source_environment':'unrecorded; analysis environment is separate'}
    errors=[];rows=[]
    if args.input_dir:
        actual={p.name for p in raw.glob('seed_*.json')}
        if actual!={f'seed_{s}.json' for s in seeds}:errors.append('unexpected or incomplete raw cohort')
    for seed in seeds:
        out=raw/f'seed_{seed}.json'
        try:
            if args.input_dir is None:
                proc=subprocess.run([sys.executable,str(RUNNER),'--seed',str(seed),'--out',str(out)],cwd=ROOT,text=True,capture_output=True,timeout=900)
                (raw/f'seed_{seed}.log').write_text(proc.stdout+'\nSTDERR:\n'+proc.stderr)
                if proc.returncode:raise RuntimeError(f'runner exit {proc.returncode}: {proc.stderr[-1000:]}')
            row=selected_row(json.loads(out.read_text()),seed,protocol)
            rows.append(row)
            print(json.dumps(row),flush=True)
        except (Exception,) as exc:
            errors.append(f'seed {seed}: {type(exc).__name__}: {exc}')
    errs,agg=compare_rows(rows,expected,protocol);errors+=errs
    report={'status':'FAIL' if errors else 'PASS','evidence_class':'reproduction_of_existing_confirmatory_cohort',
       'analysis_mode':'existing_raw_readback' if args.input_dir else 'full_runner_execution',
       'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),
       'environment':env,'analysis_environment':environment(),'seeds':seeds,'rows':rows,'aggregate':agg,'errors':errors,
       'input_sha256':inputs,'raw_sha256':{p.name:digest(p) for p in sorted(raw.iterdir()) if p.is_file()},
       'tolerances':{'budgets':'exact','log_budget_statistics_absolute':1e-12,'test_accuracy_difference_absolute':1e-6},
       'interpretation':'Strict comparison with the original summary. Directional agreement is reported separately and NEVER converts a strict mismatch into PASS. No seed replacement, threshold tuning, new confirmatory sample or independent external replication.'}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'errors':errors,'aggregate':agg}),flush=True)
    return 1 if errors else 0

if __name__=='__main__':
    raise SystemExit(main())
