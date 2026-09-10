#!/usr/bin/env python3
"""R87R1: new seed confirmation, never a replacement of the original archive."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SEEDS=tuple(range(871100,871116))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def module(name,file):
    s=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def run(seed,out,bridge=False):
    if seed not in SEEDS and not (bridge and seed==1202):raise ValueError('Outside locked seed sets')
    p=module('numeric_profile','scripts/reproduction_diagnostics/r87_profile.py')
    p.configure('portable_v1')
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled=False;torch.use_deterministic_algorithms(True)
    if torch.backends.cpu.get_cpu_capability()!='AVX2':raise RuntimeError('Effective ATen AVX2 required')
    with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON):torch.mm(torch.ones(8,8),torch.ones(8,8))
    d=module('instrumentation','scripts/reproduction_diagnostics/r87_dispatch.py')
    # Only the instrumentation admission list changes in memory; original scientific
    # runner bytes, arithmetic, training loop and selection rules stay untouched.
    d.SEEDS=(1202,) if bridge else SEEDS
    d.execute(seed,'aten_avx2',out,bridge)
    r=json.loads((out/'record.json').read_text())
    r.update(experiment='R87R1_FRESH_PROFILE_CONFIRMATION',evidence_class='IMPLEMENTATION_BRIDGE' if bridge else 'PROSPECTIVE_CONFIRMATORY',profile='portable_v1',profile_source_sha256=sha(ROOT/'scripts/reproduction_diagnostics/r87_profile.py'),fresh_harness_sha256=sha(__file__),protocol_sha256=sha(ROOT/'results/reproduction/r87r1_fresh_profile/PROTOCOL.json'))
    r['numeric_settings']={'mkldnn_enabled':torch.backends.mkldnn.enabled,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'interop_threads':torch.get_num_interop_threads(),'mkl_cbwr':os.environ.get('MKL_CBWR')}
    dump(out/'record.json',r)

def inspect(root):
    p=module('profile_validator','scripts/reproduction_diagnostics/r87_profile.py');root=Path(root)
    actual={int(x.name.split('_')[1]) for x in root.glob('seed_*') if x.is_dir()}
    if actual!=set(SEEDS):raise ValueError('Missing or unexpected seeds: '+str(actual))
    rows=[]
    for s in SEEDS:
        r,o=p.validate_record(root/f'seed_{s}')
        if r['experiment']!='R87R1_FRESH_PROFILE_CONFIRMATION' or r['seed']!=s or o['seed']!=s:raise ValueError('Mislabelled seed')
        if r['profile']!='portable_v1' or r['numeric_settings']!={'mkldnn_enabled':False,'deterministic_algorithms':True,'interop_threads':1,'mkl_cbwr':'COMPATIBLE'}:raise ValueError('Numeric profile mismatch')
        if 'CNR:COMPATIBLE' not in (root/f'seed_{s}'/'process.log').read_text():raise ValueError('Missing effective MKL probe')
        if o['selected_sep'] is None or o['selected_comp'] is None:raise ValueError('Unselected endpoint; exclusion forbidden')
        rows.append((r,o))
    return rows

def evaluate(outcomes):
    import numpy as np
    if len(outcomes)!=16 or sorted(x['seed'] for x in outcomes)!=list(SEEDS):raise ValueError('Complete unique16 required')
    os_=sorted(outcomes,key=lambda x:x['seed'])
    sep=np.array([o['selected_sep']['budget'] for o in os_],dtype=np.float64);comp=np.array([o['selected_comp']['budget'] for o in os_],dtype=np.float64)
    logs=np.log2(comp/sep);delta=np.array([o['selected_comp']['comp_test_acc']-o['selected_sep']['sep_test_acc'] for o in os_])
    if not (np.isfinite(logs).all() and np.isfinite(delta).all()):raise ValueError('Nonfinite metrics')
    ix=np.random.default_rng(87112026).integers(0,16,size=(100000,16))
    def stats(x):
        b=x[ix].mean(1);return {'mean':float(x.mean()),'median':float(np.median(x)),'ci95':np.percentile(b,[2.5,97.5]).tolist(),'bootstrap_se':float(b.std(ddof=1))}
    primary=stats(logs);secondary=stats(delta)
    primary['status']='PASS' if primary['ci95'][1]<0 else ('FAIL' if primary['ci95'][0]>0 else 'UNCERTAIN')
    secondary['status']='PASS' if secondary['ci95'][0]>-.02 else ('FAIL' if secondary['ci95'][1]<-.02 else 'UNCERTAIN')
    success=primary['status']=='PASS' and secondary['status']=='PASS'
    return {'experiment':'R87R1_FRESH_PROFILE_CONFIRMATION','evidence_class':'PROSPECTIVE_CONFIRMATORY','decision':'R87R1_CONFIRMATORY_PASS' if success else ('R87R1_CONFIRMATORY_FAIL' if 'FAIL' in (primary['status'],secondary['status']) else 'R87R1_CONFIRMATORY_UNCERTAIN'),'attempted':16,'eligible':16,'primary_log2_budget_ratio':primary,'secondary_test_accuracy_difference':secondary,'mean_componentwise_budget':float(sep.mean()),'mean_composed_budget':float(comp.mean()),'geometric_budget_ratio':float(2**logs.mean()),'geometric_budget_ratio_ci95':[float(2**v) for v in primary['ci95']],'composed_lower_count':int((comp<sep).sum()),'bootstrap_resamples':100000,'bootstrap_seed':87112026,'test_evaluated':True,'test_used_for_selection':False,'independent_dataset':False,'historical_archive_recovered':False}

def aggregate(a,b,out):
    aa,bb=inspect(a),inspect(b);pairs=[]
    import numpy as np
    for (ra,oa),(rb,ob) in zip(aa,bb):
        for k in ('profile_source_sha256','fresh_harness_sha256','protocol_sha256'):
            if ra[k]!=rb[k]:raise ValueError('Source/protocol mismatch')
        controls=all(ra['trace'][k]==rb['trace'][k] for k in ('dataset_hashes','random_stream_hashes','random_stream_calls')) and ra['trace']['teacher']['initial_state']==rb['trace']['teacher']['initial_state']
        s=oa['seed'];xa=np.load(Path(a)/f'seed_{s}/trace.npz',allow_pickle=False);xb=np.load(Path(b)/f'seed_{s}/trace.npz',allow_pickle=False)
        pairs.append({'seed':s,'controls_identical':controls,'outcome_exact':oa==ob,'trace_ledger_exact':ra['trace']==rb['trace'],'trace_arrays_exact':set(xa.files)==set(xb.files) and all(np.array_equal(xa[k],xb[k]) for k in xa.files)})
    decision=evaluate([o for r,o in aa]);second=evaluate([o for r,o in bb])
    technical=all(all(v for k,v in x.items() if k!='seed') for x in pairs)
    decision['technical_replication']='TESTED_HOST_EXACT_REPLICATION_PASS' if technical else 'TECHNICAL_REPLICATION_FAIL'
    decision['host_b_scientific_decision']=second['decision']
    dump(Path(out)/'DECISION.json',decision);dump(Path(out)/'FRESH_ROWS.json',{'host_a':[o for r,o in aa],'host_b':[o for r,o in bb]});dump(Path(out)/'CROSS_HOST_AUDIT.json',{'pairs':pairs,'status':'PASS' if technical else 'FAIL','scope':'Implementation/numeric reproducibility, not independent scientific peer review'})
    print(json.dumps(decision),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('seed','aggregate'),required=True);ap.add_argument('--seed',type=int);ap.add_argument('--bridge',action='store_true');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--host-a',type=Path);ap.add_argument('--host-b',type=Path);a=ap.parse_args()
    if a.mode=='seed':run(a.seed,a.out,a.bridge)
    else:aggregate(a.host_a,a.host_b,a.out)
