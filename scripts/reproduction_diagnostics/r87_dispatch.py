#!/usr/bin/env python3
"""Known-cohort diagnostic. Original science runner remains byte-identical."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, platform, subprocess, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'scripts/reproduce/core_discovery_digits/run_confirmatory.py'
EXPECTED_BLOB = '8759933ed2bb95014c3afc835acafa1743e40ea6'
SEEDS = tuple(range(1200, 1208))

def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def tensor_hash(t):
    a=t.detach().cpu().contiguous().numpy()
    return digest(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes())

def model_hash(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():
        h.update(k.encode()); h.update(tensor_hash(v).encode())
    return h.hexdigest()

def flat(m, grad=False):
    return torch.cat([(p.grad if grad else p).detach().flatten().cpu() for p in m.parameters()]).numpy().copy()

def load_runner():
    raw=RUNNER.read_bytes()
    b=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if b!=EXPECTED_BLOB: raise ValueError(f'original science source mismatch: {b}')
    s=importlib.util.spec_from_file_location('original_core', RUNNER)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m

def metadata():
    import importlib.metadata as im
    try: cpu=subprocess.check_output(['lscpu'],text=True)
    except (FileNotFoundError,subprocess.SubprocessError): cpu=platform.processor()
    return dict(python=sys.version, packages={k:im.version(k) for k in ('torch','numpy','scikit-learn','scipy')},
                cpu=cpu, platform=platform.platform(), torch_build=torch.__config__.show(),
                threads=torch.get_num_threads(), aten_capability=torch.backends.cpu.get_cpu_capability(),
                environment={k:os.environ.get(k) for k in ('ATEN_CPU_CAPABILITY','MKL_CBWR','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
                source_sha256=digest(RUNNER.read_bytes()), instrumentation_sha256=digest(Path(__file__).read_bytes()),
                github_sha=os.environ.get('GITHUB_SHA'), github_run_id=os.environ.get('GITHUB_RUN_ID'))

def execute(seed, mode, out: Path, bridge=False):
    if seed not in SEEDS: raise ValueError('Only the locked known cohort is allowed')
    r=load_runner(); torch.set_num_threads(1)
    if mode=='aten_avx2' and torch.backends.cpu.get_cpu_capability()!='AVX2':
        raise ValueError('AVX2 intervention was not applied before torch import')
    data=r.datasets()
    traces={}; info={'dataset_hashes':{k:tensor_hash(v) for k,v in zip(('Xt','yt','Xv','yv','Xte','yte'),data)}, 'teacher':{},'fits':[]}
    orig_train=r.train_model; orig_fit=r.fit_map; orig_joint=r.fit_joint
    streams={'randperm':hashlib.sha256(),'randint':hashlib.sha256()}; calls={'randperm':0,'randint':0}
    originals={name:getattr(torch,name) for name in streams}
    def tracked_rng(name):
        def wrapped(*a,**kw):
            x=originals[name](*a,**kw); streams[name].update(tensor_hash(x).encode()); calls[name]+=1; return x
        return wrapped
    # Instrument only the teacher loop; every arithmetic operation matches the original.
    def train(seed, Xt, yt, epochs=60):
        model=r.Net(seed)
        info['teacher']['initial_state']=model_hash(model)
        traces['initial_parameters']=flat(model)
        opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
        gen=torch.Generator().manual_seed(seed+999); step=0
        for epoch in range(epochs):
            perm=torch.randperm(len(Xt),generator=gen)
            for i in range(0,len(Xt),64):
                ix=perm[i:i+64]; opt.zero_grad()
                logits=model(Xt[ix]); loss=r.F.cross_entropy(logits,yt[ix]); loss.backward()
                if step==0:
                    traces['first_batch_indices']=ix.numpy().copy()
                    traces['first_logits']=logits.detach().numpy().copy()
                    traces['first_gradients']=flat(model,grad=True)
                    info['teacher']['first_loss']=float(loss.detach())
                opt.step(); step+=1
                if step in (1,2,10,17,170,1020):
                    traces[f'parameters_step_{step}']=flat(model)
                    info['teacher'][f'state_step_{step}']=model_hash(model)
        info['teacher']['final_state']=model_hash(model)
        info['teacher']['steps']=step
        return model
    def fit(module,X,Y,updates,seed):
        rec={'fit_seed':seed,'updates':updates,'initial_state':model_hash(module), 'input_hash':tensor_hash(X),'target_hash':tensor_hash(Y)}
        m=orig_fit(module,X,Y,updates,seed); rec['final_state']=model_hash(m); info['fits'].append(rec); return m
    def joint(a,b,X,Y,updates,seed):
        ret=orig_joint(a,b,X,Y,updates,seed); info['joint_final']=[model_hash(x) for x in ret]; return ret
    r.train_model=train; r.fit_map=fit; r.fit_joint=joint
    for name in streams: setattr(torch,name,tracked_rng(name))
    try: outcome=r.run(seed)
    finally:
        for name,f in originals.items():setattr(torch,name,f)
        r.train_model=orig_train; r.fit_map=orig_fit; r.fit_joint=orig_joint
    info['random_stream_hashes']={k:v.hexdigest() for k,v in streams.items()}
    info['random_stream_calls']=calls
    if bridge:
        reference=r.run(seed)
        if reference!=outcome: raise AssertionError('instrumentation changes original outcome')
        info['bridge_original_outcome_exact']=True
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'trace.npz',**traces)
    (out/'outcome.json').write_text(json.dumps(outcome,indent=2)+'\n')
    record={'experiment':'R87D1_CPU_DISPATCH_DIAGNOSTIC','evidence_class':'DIAGNOSTIC_KNOWN_SEED_NOT_FRESH','seed':seed,'mode':mode,
            'metadata':metadata(),'trace':info,'outcome_sha256':digest((out/'outcome.json').read_bytes()),'trace_npz_sha256':digest((out/'trace.npz').read_bytes())}
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'seed':seed,'mode':mode,'capability':record['metadata']['aten_capability'],'sep':outcome['selected_sep']['budget'] if outcome['selected_sep'] else None,'comp':outcome['selected_comp']['budget'] if outcome['selected_comp'] else None,'bridge':bridge}),flush=True)

def summarize(root: Path, out: Path):
    records={}
    for p in root.glob('*/record.json'):
        r=json.loads(p.read_text()); outcome=json.loads((p.parent/'outcome.json').read_text())
        if r['outcome_sha256']!=digest((p.parent/'outcome.json').read_bytes()):raise AssertionError('outcome hash')
        if r['trace_npz_sha256']!=digest((p.parent/'trace.npz').read_bytes()):raise AssertionError('trace hash')
        records[p.parent.name]=(r,outcome)
    expected=[f'{s}_{m}' for s in SEEDS for m in ('native_default','aten_avx2')]
    missing=[k for k in expected if k not in records]
    comparisons=[]; arrays={}; budget_changed=[]; numeric_changed=[]
    for s in SEEDS:
        keys=[f'{s}_{m}' for m in ('native_default','aten_avx2')]
        if any(k not in records for k in keys):continue
        (a,ao),(b,bo)=[records[k] for k in keys]
        controls=all(a['trace'][k]==b['trace'][k] for k in ('dataset_hashes','random_stream_hashes','random_stream_calls')) and a['trace']['teacher']['initial_state']==b['trace']['teacher']['initial_state']
        pa=np.load(root/keys[0]/'trace.npz',allow_pickle=False); pb=np.load(root/keys[1]/'trace.npz',allow_pickle=False)
        diffs={}
        for name in pa.files:
            xa,xb=pa[name].astype(np.float64),pb[name].astype(np.float64)
            diffs[name]={'exact':bool(np.array_equal(xa,xb)),'max_abs':float(np.max(np.abs(xa-xb))), 'l2':float(np.linalg.norm(xa-xb))}
        endpoint=lambda o:[None if o[k] is None else o[k]['budget'] for k in ('selected_sep','selected_comp')]
        change=endpoint(ao)!=endpoint(bo)
        if change:budget_changed.append(s)
        if ao!=bo:numeric_changed.append(s)
        comparisons.append({'seed':s,'controls_identical':controls,'native_capability':a['metadata']['aten_capability'],'avx2_capability':b['metadata']['aten_capability'],
                            'native_budgets':endpoint(ao),'avx2_budgets':endpoint(bo),'outcomes_exact':ao==bo,'budget_changed':change,'stage_differences':diffs})
    repeats=[]
    for s in (1202,1205):
        for m in ('native_default','aten_avx2'):
            k=f'{s}_{m}';kr=k+'_repeat'
            if k in records and kr in records:
                a,ao=records[k];b,bo=records[kr]
                repeats.append({'seed':s,'mode':m,'outcome_exact':ao==bo,'traces_exact':a['trace']==b['trace']})
    archive=json.loads((ROOT/'results/core_discovery_digits/confirm_summary.json').read_text())
    ar={x['seed']:x for x in archive['per_seed_selected_budgets']}
    cohorts={}
    for m in ('native_default','aten_avx2'):
        os_=[records[f'{s}_{m}'][1] for s in SEEDS if f'{s}_{m}' in records]
        if len(os_)!=8 or any(x['selected_sep'] is None or x['selected_comp'] is None for x in os_):continue
        sep=np.array([x['selected_sep']['budget'] for x in os_]);comp=np.array([x['selected_comp']['budget'] for x in os_]);logs=np.log2(comp/sep)
        delta=np.array([x['test_acc_diff_comp_minus_sep'] for x in os_]);ix=np.random.default_rng(991200).integers(0,8,size=(20000,8))
        mismatches=[]
        for x in os_:
            a=ar[x['seed']]
            if x['selected_sep']['budget']!=a['componentwise'] or x['selected_comp']['budget']!=a['composed'] or abs(x['test_acc_diff_comp_minus_sep']-a['test_acc_diff'])>1e-6:
                mismatches.append(x['seed'])
        cohorts[m]={'mean_sep':float(sep.mean()),'mean_comp':float(comp.mean()),'geometric_ratio':float(2**logs.mean()),'log2_ci95':np.percentile(logs[ix].mean(1),[2.5,97.5]).tolist(),'test_difference_ci95':np.percentile(delta[ix].mean(1),[2.5,97.5]).tolist(), 'composed_lower_count':int((comp<sep).sum()),'strict_archive_mismatch_seeds':mismatches}
    caps_differ=any(x['native_capability']!=x['avx2_capability'] for x in comparisons)
    if missing or not all(x['controls_identical'] for x in comparisons): decision='STOP_INTEGRITY_OR_INPUT_CONTROL'
    elif not caps_differ:decision='NO_CAPABILITY_CONTRAST_NEGATIVE_CONTROL'
    elif budget_changed:decision='DISPATCH_ENDPOINT_SENSITIVITY_OBSERVED'
    elif numeric_changed:decision='DISPATCH_NUMERIC_SENSITIVITY_ONLY'
    else:decision='DISPATCH_SENSITIVITY_NOT_OBSERVED'
    report={'experiment':'R87D1_CPU_DISPATCH_DIAGNOSTIC','decision':decision,'missing':missing,'primary_comparisons':comparisons,'budget_changed_seeds':budget_changed,'numeric_changed_seeds':numeric_changed,'repeats':repeats,'cohorts':cohorts,'record_count':len(records),'historical_cause_resolved':False,'new_independent_confirmation':False}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('decision','missing','budget_changed_seeds','repeats','cohorts','record_count')}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int);ap.add_argument('--mode',choices=('native_default','aten_avx2'));ap.add_argument('--out',type=Path,required=True);ap.add_argument('--bridge',action='store_true');ap.add_argument('--summarize',type=Path);a=ap.parse_args()
    if a.summarize:summarize(a.summarize,a.out)
    else:execute(a.seed,a.mode,a.out,a.bridge)
