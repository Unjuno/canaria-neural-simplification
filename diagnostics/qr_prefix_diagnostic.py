"""Post-hoc numerical mechanism diagnostic; not a fresh confirmatory cohort."""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, os
from pathlib import Path
import numpy as np
import torch
ROOT=Path(os.environ.get('CANARIA_REPO',str(Path(__file__).resolve().parent.parent/'canaria_work')))
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.gaussian_shift_interface.run_c61r_seed import build_base_hierarchy,canonical_nested_qr
from scripts.gaussian_shift_interface.run_c68e_seed import train_paired_teachers,state_dict_sha256
from scripts.gaussian_shift_interface.run_c72e_seed import nested_calibration_indices,standard_normal_like,shifted_input
from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import split_data,acts
SEEDS=(69300,70300,71300)
DIMS=(1,2,4,8,16,32,64)
def projector(q,k): return q[:,:k]@q[:,:k].T

def inspect(rc,rv):
    q192=canonical_nested_qr(rc[:192]);q384=canonical_nested_qr(rc)
    gen=torch.Generator().manual_seed(739001)
    perms={'identity':torch.arange(len(rc)), 'reverse':torch.arange(len(rc)-1,-1,-1),
           'random1':torch.randperm(len(rc),generator=gen), 'random2':torch.randperm(len(rc),generator=gen)}
    qs={name:canonical_nested_qr(rc[ix]) for name,ix in perms.items()}
    vd=torch.linalg.svd(rc.double(),full_matrices=False)[2].T
    vr=torch.linalg.svd(rc[perms['reverse']].double(),full_matrices=False)[2].T
    records={}
    for k in DIMS:
        p=projector(q384,k); prefix=canonical_nested_qr(rc[:k])
        values={'append_projector_fro_diff':float(torch.linalg.matrix_norm(projector(q192,k)-p)),
                'prefix_only_projector_fro_diff':float(torch.linalg.matrix_norm(projector(prefix,k)-p)),
                'first_k_residual_rank':int(torch.linalg.matrix_rank(rc[:k].double())),
                'svd_reverse_projector_fro_diff_float64':float(torch.linalg.matrix_norm(projector(vd,k)-projector(vr,k))),
                'orders':{}}
        for name,q in qs.items():
            pp=projector(q,k)
            values['orders'][name]={'projector_fro_diff_from_identity':float(torch.linalg.matrix_norm(pp-p)),
                'validation_residual_energy_capture':float(((rv@q[:,:k])**2).sum()/(rv**2).sum())}
        records[str(k)]=values
    return records

def run(seed,out):
    if seed not in SEEDS: raise ValueError('only designated diagnostic verification seeds')
    torch.set_num_threads(1)
    xt,yt,xv,yv=split_data()
    _,teacher,prov=train_paired_teachers(seed,xt,yt)
    base,budget=build_base_hierarchy(seed,*acts(teacher,xt))
    bi,ei,li=nested_calibration_indices(len(xt))
    xb=xt[torch.as_tensor(bi)];xe=xt[torch.as_tensor(ei)]
    xb=shifted_input(xb,standard_normal_like(xb,seed+710100))
    xe=shifted_input(xe,standard_normal_like(xe,seed+720101))
    xs=shifted_input(xv,standard_normal_like(xv,seed+710200))
    ac=acts(teacher,torch.cat([xb,xe]));av=acts(teacher,xs)
    with torch.no_grad(): rc=ac[-1]-base(ac[0]);rv=av[-1]-base(av[0])
    result={'kind':'POST_HOC_MECHANISM_DIAGNOSTIC','not_fresh':True,'seed':seed,
            'dataset':'digits fixed split','architecture':'Residual-MLP robust teacher',
            'sigma':.36,'calibration_samples':384,'validation_examples':270,'test_evaluated':False,
            'teacher_state_sha256':state_dict_sha256(teacher),'base_state_sha256':state_dict_sha256(base),
            'metrics':inspect(rc,rv),
            'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'threads':1},
            'interpretation':'Construction diagnostic only. Projected validation residual uses teacher oracle, not deployable task accuracy. Different local versions are not pooled into GitHub cohorts.'}
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/f'residuals_{seed}.npz',calibration=rc.numpy(),validation=rv.numpy())
    result['residual_npz_sha256']=hashlib.sha256((out/f'residuals_{seed}.npz').read_bytes()).hexdigest()
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/f'seed_{seed}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(seed,'k2',result['metrics']['2'],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();run(a.seed,a.out)
