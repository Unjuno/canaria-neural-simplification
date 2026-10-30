#!/usr/bin/env python3
"""Fresh confirmation under an explicitly new numerical recipe, not old recovery."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SEEDS=tuple(range(871200,871216))
EXP='R87R2_FRESH_SQRT_PROFILE_CONFIRMATION'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def load(name,file):
    s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def execute(seed,out,bridge=False):
    if seed not in SEEDS and not (bridge and seed==1202):raise ValueError('Outside locked admission')
    p=load('baseprofile','scripts/reproduction_diagnostics/r87_profile.py');p.configure('portable_v1')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mkldnn.enabled=False;torch.use_deterministic_algorithms(True)
    if torch.backends.cpu.get_cpu_capability()!='AVX2':raise RuntimeError('Effective AVX2 missing')
    sm=load('sqrtprofile','scripts/reproduction_diagnostics/r87_sqrt_profile.py');counts,restore=sm.install_scoped_sqrt64(torch)
    try:
        with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON):torch.mm(torch.ones(8,8),torch.ones(8,8))
        d=load('instrument','scripts/reproduction_diagnostics/r87_dispatch.py');d.SEEDS=(1202,) if bridge else SEEDS
        d.execute(seed,'aten_avx2',out,bridge)
    finally:restore()
    if counts['float32_sqrt_calls']<=0:raise RuntimeError('Intervention inactive')
    r=json.loads((out/'record.json').read_text());r.update(experiment=EXP,evidence_class='IMPLEMENTATION_BRIDGE' if bridge else 'PROSPECTIVE_CONFIRMATORY',profile='portable_v2_sqrt64',sqrt_intervention=counts)
    r['source_hashes']={x:sha(ROOT/x) for x in ('scripts/reproduction_diagnostics/r87r2_confirm.py','scripts/reproduction_diagnostics/r87_profile.py','scripts/reproduction_diagnostics/r87_sqrt_profile.py','scripts/reproduction_diagnostics/r87_dispatch.py','scripts/reproduce/core_discovery_digits/run_confirmatory.py','results/reproduction/r87r2_fresh_sqrt_profile/PROTOCOL.json')}
    r['numeric_settings']={'mkldnn':False,'deterministic_algorithms':True,'threads':1,'interop_threads':1,'mkl_cbwr':'COMPATIBLE','capability':'AVX2'}
    dump(out/'record.json',r)

def validate(root):
    root=Path(root);p=load('validator','scripts/reproduction_diagnostics/r87_profile.py')
    dirs=sorted(root.glob('seed_*'))
    if {x.name for x in dirs}!={f'seed_{s}' for s in SEEDS}:raise ValueError('Missing/extra seed directories')
    records=[]
    for s in SEEDS:
        r,o=p.validate_record(root/f'seed_{s}')
        if r['experiment']!=EXP or r['seed']!=s or o['seed']!=s or r['evidence_class']!='PROSPECTIVE_CONFIRMATORY':raise ValueError('Wrong experiment/seed')
        if r['profile']!='portable_v2_sqrt64' or r['sqrt_intervention']['float32_sqrt_calls']<=0:raise ValueError('Inactive profile')
        if r['numeric_settings']!={'mkldnn':False,'deterministic_algorithms':True,'threads':1,'interop_threads':1,'mkl_cbwr':'COMPATIBLE','capability':'AVX2'}:raise ValueError('Incorrect numeric settings')
        if 'CNR:COMPATIBLE' not in (root/f'seed_{s}'/'process.log').read_text():raise ValueError('No effective MKL probe')
        records.append((r,o))
    return records

def evaluate(outcomes):
    import numpy as np
    if len(outcomes)!=16 or sorted(x['seed'] for x in outcomes)!=list(SEEDS):raise ValueError('Complete unique16 required')
    oo=sorted(outcomes,key=lambda o:o['seed'])
    if any(o.get('selected_sep') is None or o.get('selected_comp') is None for o in oo):raise ValueError('Unselected endpoint; exclusion forbidden')
    sep=np.array([o['selected_sep']['budget'] for o in oo],dtype=float);comp=np.array([o['selected_comp']['budget'] for o in oo],dtype=float)
    delta=np.array([o['selected_comp']['comp_test_acc']-o['selected_sep']['sep_test_acc'] for o in oo]);logs=np.log2(comp/sep)
    if not (np.isfinite(logs).all() and np.isfinite(delta).all() and (sep>0).all() and (comp>0).all()):raise ValueError('Nonfinite or invalid metric')
    ix=np.random.default_rng(87122026).integers(0,16,size=(100000,16))
    def stats(x):
        b=x[ix].mean(1);return {'mean':float(x.mean()),'median':float(np.median(x)),'ci95':np.percentile(b,[2.5,97.5]).tolist(),'bootstrap_se':float(b.std(ddof=1))}
    p=stats(logs);s=stats(delta)
    p['status']='PASS' if p['ci95'][1]<0 else ('FAIL' if p['ci95'][0]>0 else 'UNCERTAIN')
    s['status']='PASS' if s['ci95'][0]>-.02 else ('FAIL' if s['ci95'][1]<-.02 else 'UNCERTAIN')
    status='PASS' if p['status']==s['status']=='PASS' else ('FAIL' if 'FAIL' in (p['status'],s['status']) else 'UNCERTAIN')
    return {'experiment':EXP,'evidence_class':'PROSPECTIVE_CONFIRMATORY','decision':'R87R2_CONFIRMATORY_'+status,'attempted':16,'eligible':16,'primary_log2_budget_ratio':p,'secondary_test_accuracy_difference':s,'mean_componentwise_budget':float(sep.mean()),'mean_composed_budget':float(comp.mean()),'geometric_budget_ratio':float(2**logs.mean()),'geometric_budget_ratio_ci95':[float(2**v) for v in p['ci95']],'composed_lower_count':int((comp<sep).sum()),'teacher_val_mean':float(np.mean([o['teacher_val_acc'] for o in oo])),'teacher_test_mean':float(np.mean([o['teacher_test_acc'] for o in oo])),'bootstrap_resamples':100000,'bootstrap_seed':87122026,'test_evaluated':True,'test_used_for_selection':False,'independent_dataset':False,'historical_archive_recovered':False}

def suite(root,workers):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    def one(seed,name,bridge=False):
        out=root/name;out.mkdir(exist_ok=True);cmd=[sys.executable,__file__,'--mode','seed','--seed',str(seed),'--out',str(out)]+(['--bridge'] if bridge else [])
        with (out/'process.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=360)
        dump(out/'EXIT.json',{'returncode':r.returncode,'argv':cmd})
        if r.returncode:raise RuntimeError('Failed '+name)
        print('COMPLETE',name,flush=True)
    one(1202,'preflight',True)
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fs={pool.submit(one,s,f'seed_{s}'):s for s in SEEDS}
        for future in concurrent.futures.as_completed(fs):
            try:future.result()
            except Exception as ex:errors.append({'seed':fs[future],'error':repr(ex)})
    dump(root/'EXECUTION_SUMMARY.json',{'planned_seeds':list(SEEDS),'errors':errors,'worker_processes':workers,'independent_model_seed_count':16,'technical_repeats_are_not_new_seeds':True})
    if errors:raise RuntimeError('Execution failures retained, no exclusions')
    records=validate(root);decision=evaluate([o for r,o in records]);dump(root/'DECISION.json',decision)
    dump(root/'FILES_SHA256.json',{str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name!='FILES_SHA256.json'})
    print(json.dumps(decision),flush=True)

def selftest():
    import copy
    oo=[{'seed':s,'selected_sep':{'budget':4096,'sep_test_acc':.95},'selected_comp':{'budget':2048,'comp_test_acc':.95},'teacher_val_acc':.98,'teacher_test_acc':.98} for s in SEEDS]
    assert evaluate(oo)['decision']=='R87R2_CONFIRMATORY_PASS'
    x=copy.deepcopy(oo)
    for o in x:o['selected_comp']['comp_test_acc']=.90
    assert evaluate(x)['decision']=='R87R2_CONFIRMATORY_FAIL'
    x=copy.deepcopy(oo)
    for o in x:o['selected_comp']['budget']=4096
    assert evaluate(x)['decision']=='R87R2_CONFIRMATORY_UNCERTAIN'
    bad=[oo[:-1],oo[:-1]+[oo[0]]]
    x=copy.deepcopy(oo);x[0]['selected_comp']=None;bad.append(x)
    x=copy.deepcopy(oo);x[0]['selected_comp']['comp_test_acc']=float('nan');bad.append(x)
    for x in bad:
        try:evaluate(x)
        except ValueError:pass
        else:raise AssertionError('Malformed cohort accepted')
    print('R87R2 EVALUATOR SYNTHETIC TESTS PASS',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('seed','suite','selftest'),required=True);p.add_argument('--seed',type=int);p.add_argument('--bridge',action='store_true');p.add_argument('--out',type=Path);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.mode=='seed':execute(a.seed,a.out,a.bridge)
    elif a.mode=='suite':suite(a.out,a.workers)
    else:selftest()
