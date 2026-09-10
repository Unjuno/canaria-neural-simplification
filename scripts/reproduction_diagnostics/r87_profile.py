#!/usr/bin/env python3
"""R87D5: execute separately labelled numeric profiles; never patch old source."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(1200,1208))
PROFILES = ('avx2_baseline','portable_v1')
ENV = {'ATEN_CPU_CAPABILITY':'avx2','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','MKL_DYNAMIC':'FALSE','OMP_DYNAMIC':'FALSE'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,obj):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def configure(profile):
    if profile not in PROFILES: raise ValueError(profile)
    os.environ.update(ENV)
    os.environ.pop('MKL_CBWR',None)
    if profile=='portable_v1': os.environ['MKL_CBWR']='COMPATIBLE'

def execute(seed,profile,out,bridge=False):
    if seed not in SEEDS: raise ValueError('Known-cohort diagnostic only')
    configure(profile)
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled=profile=='avx2_baseline'
    torch.use_deterministic_algorithms(profile=='portable_v1')
    if torch.backends.cpu.get_cpu_capability()!='AVX2': raise RuntimeError('Effective ATen AVX2 not established')
    spec=importlib.util.spec_from_file_location('r87_dispatch',ROOT/'scripts/reproduction_diagnostics/r87_dispatch.py')
    d=importlib.util.module_from_spec(spec); spec.loader.exec_module(d)
    print('MKL effective-branch probe follows',flush=True)
    with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON):
        torch.mm(torch.ones(8,8),torch.ones(8,8))
    d.execute(seed,'aten_avx2',out,bridge)
    record=json.loads((out/'record.json').read_text())
    record.update(experiment='R87D5_PORTABLE_NUMERIC_PROFILE',profile=profile)
    record['profile_settings']={'environment':{k:os.environ.get(k) for k in (*ENV,'MKL_CBWR')},'mkldnn_enabled':torch.backends.mkldnn.enabled,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'interop_threads':torch.get_num_interop_threads(),'float32_matmul_precision':torch.get_float32_matmul_precision()}
    record['profile_source_sha256']=sha(__file__)
    dump(out/'record.json',record)

def validate_record(path):
    import numpy as np
    p=Path(path); r=json.loads((p/'record.json').read_text()); o=json.loads((p/'outcome.json').read_text())
    if sha(p/'outcome.json')!=r['outcome_sha256']: raise AssertionError('outcome hash')
    if (p/'trace.npz').exists() and sha(p/'trace.npz')!=r['trace_npz_sha256']: raise AssertionError('trace hash')
    teacher=o['teacher_val_acc']
    for row in o['grid']:
        for side in ('sep','comp'):
            if not all(np.isfinite(row[k]) for k in (side+'_nmse',side+'_val_acc')): raise ValueError('nonfinite')
            gate=row[side+'_nmse']<=0.08 and row[side+'_val_acc']>=teacher-0.02
            if gate!=row[side+'_pass']: raise AssertionError('candidate gate')
    for side in ('sep','comp'):
        passing=[x for x in o['grid'] if x[side+'_pass']]
        selected=o['selected_'+side]
        if bool(passing)!=(selected is not None): raise AssertionError('selection availability')
        if passing and selected['budget']!=passing[0]['budget']: raise AssertionError('nonminimal selected grid point')
    return r,o

def summarize(root):
    import numpy as np
    root=Path(root); results={}; missing=[]
    for s in SEEDS:
        for p in PROFILES:
            key=f'{s}_{p}'
            if not (root/key/'record.json').exists(): missing.append(key);continue
            results[key]=validate_record(root/key)
    repeats=[]
    for s in (1202,1205):
        for p in PROFILES:
            key=f'{s}_{p}'; rep=key+'_repeat'
            if not (root/rep/'record.json').exists(): missing.append(rep);continue
            b,bo=validate_record(root/rep)
            if key in results:
                a,ao=results[key]; repeats.append({'seed':s,'profile':p,'outcome_exact':ao==bo,'traces_exact':a['trace']==b['trace']})
    cohorts={}
    archive=json.loads((ROOT/'results/core_discovery_digits/confirm_summary.json').read_text())
    ar={int(x['seed']):x for x in archive['per_seed_selected_budgets']}
    for p in PROFILES:
        entries=[results[f'{s}_{p}'] for s in SEEDS if f'{s}_{p}' in results]
        if len(entries)!=8:continue
        outs=[o for _,o in entries]
        complete=all(o['selected_sep'] and o['selected_comp'] for o in outs)
        if complete:
            sep=np.array([o['selected_sep']['budget'] for o in outs]); comp=np.array([o['selected_comp']['budget'] for o in outs])
            mismatch=[o['seed'] for o in outs if o['selected_sep']['budget']!=ar[o['seed']]['componentwise'] or o['selected_comp']['budget']!=ar[o['seed']]['composed'] or abs(o['test_acc_diff_comp_minus_sep']-ar[o['seed']]['test_acc_diff'])>1e-6]
            cohorts[p]={'mean_componentwise':float(sep.mean()),'mean_composed':float(comp.mean()),'geometric_ratio':float(np.exp(np.log(comp/sep).mean())),'composed_lower_count':int((comp<sep).sum()),'archive_mismatch_seeds':mismatch,'outcome_sha256':{str(s):results[f'{s}_{p}'][0]['outcome_sha256'] for s in SEEDS}}
    pairs=[]
    for s in SEEDS:
        if all(f'{s}_{p}' in results for p in PROFILES):
            (a,ao),(b,bo)=[results[f'{s}_{p}'] for p in PROFILES]
            ctrl=all(a['trace'][k]==b['trace'][k] for k in ('dataset_hashes','random_stream_hashes','random_stream_calls')) and a['trace']['teacher']['initial_state']==b['trace']['teacher']['initial_state']
            pairs.append({'seed':s,'data_init_rng_identical':ctrl,'outcome_exact':ao==bo,'teacher_state_exact':a['trace']['teacher']==b['trace']['teacher'],'fits_exact':a['trace']['fits']==b['trace']['fits']})
    integrity=not missing and len(repeats)==4 and all(x['data_init_rng_identical'] for x in pairs) and all(x['outcome_exact'] and x['traces_exact'] for x in repeats)
    summary={'experiment':'R87D5_PORTABLE_NUMERIC_PROFILE','within_host_integrity':'PASS' if integrity else 'FAIL','missing':missing,'repeats':repeats,'cohorts':cohorts,'pairs':pairs,'cross_host_validation':'NOT_PERFORMED_BY_THIS_HOST_SUMMARY','historical_reproduction_resolved':False}
    dump(root/'SUMMARY.json',summary)
    return summary

def suite(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    def run_one(seed,profile,name,bridge=False):
        out=root/name;out.mkdir(exist_ok=True)
        cmd=[sys.executable,__file__,'--mode','seed','--seed',str(seed),'--profile',profile,'--out',str(out)]
        if bridge:cmd+=['--bridge']
        with (out/'process.log').open('w') as f:
            p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=300)
        dump(out/'EXIT.json',{'returncode':p.returncode,'argv':cmd})
        if p.returncode: raise RuntimeError('record failed: '+name)
        print('COMPLETED',name,flush=True)
    # Fail before the primary suite if the instrumented/original bridge is not exact.
    run_one(1202,'portable_v1','preflight',True)
    for s in SEEDS:
        for p in PROFILES:run_one(s,p,f'{s}_{p}')
    for s in (1202,1205):
        for p in PROFILES:run_one(s,p,f'{s}_{p}_repeat')
    summary=summarize(root)
    dump(root/'FILES_SHA256.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='FILES_SHA256.json'})
    print(json.dumps(summary),flush=True)
    if summary['within_host_integrity']!='PASS':raise RuntimeError('suite integrity failure')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('seed','suite','summarize'),required=True);ap.add_argument('--seed',type=int);ap.add_argument('--profile',choices=PROFILES);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--bridge',action='store_true');a=ap.parse_args()
    if a.mode=='seed':execute(a.seed,a.profile,a.out,a.bridge)
    elif a.mode=='suite':suite(a.out)
    else: print(json.dumps(summarize(a.out)))
