#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, platform, sys
from pathlib import Path

# Process-level controls must precede torch import.
os.environ.setdefault('ATEN_CPU_CAPABILITY','avx2')
os.environ.setdefault('MKL_CBWR','COMPATIBLE')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_DYNAMIC','FALSE')
os.environ.setdefault('OMP_DYNAMIC','FALSE')

import numpy as np
import sklearn
import torch

ROOT=Path(__file__).resolve().parents[3]
BASE_RUNNER=ROOT/'scripts/phase3/regression_external_validity/run_seed.py'
PROTOCOL=ROOT/'results/phase3b/stronger_teacher_regression/STAGE_B_CONFIRMATORY_PROTOCOL.json'

def sha256(path: Path) -> str:
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def state_sha256(model) -> str:
    h=hashlib.sha256()
    for name,t in model.state_dict().items():
        a=t.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def load_base():
    spec=importlib.util.spec_from_file_location('phase3_base_runner',BASE_RUNNER)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def install_scoped_sqrt64():
    original_step=torch.optim.AdamW.step
    original_sqrt=torch.Tensor.sqrt
    counts={'float32_sqrt_calls':0,'adamw_steps':0}
    def stable_sqrt(t):
        if t.dtype==torch.float32 and t.device.type=='cpu':
            counts['float32_sqrt_calls']+=1
            return original_sqrt(t.to(torch.float64)).to(torch.float32)
        return original_sqrt(t)
    def step(self,*a,**kw):
        if torch.Tensor.sqrt is not original_sqrt:
            raise RuntimeError('Unexpected concurrent Tensor.sqrt override')
        counts['adamw_steps']+=1
        torch.Tensor.sqrt=stable_sqrt
        try:return original_step(self,*a,**kw)
        finally:torch.Tensor.sqrt=original_sqrt
    torch.optim.AdamW.step=step
    def restore():
        torch.optim.AdamW.step=original_step;torch.Tensor.sqrt=original_sqrt
    return counts,restore

def configure_numeric_profile():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled=False
    torch.use_deterministic_algorithms(True)
    if torch.backends.cpu.get_cpu_capability()!='AVX2':
        raise RuntimeError('ATen AVX2 profile not active')

def main(seed:int,out:Path):
    protocol=json.loads(PROTOCOL.read_text())
    if seed not in protocol['fresh_model_seeds'] and seed!=2399:
        raise ValueError('seed outside locked fresh/preflight sets')
    configure_numeric_profile();base=load_base();counts,restore=install_scoped_sqrt64()
    try:
        # Candidate replacement experiment. The original Phase3 implementation is reused;
        # only teacher_epochs changes to the Stage-A-selected value and the numerical profile
        # wraps AdamW steps.
        result=base.run(seed,nmse_threshold=0.12,r2_tolerance=0.05,
                        grid=(2,4,6,8,12,16,20,24,32),teacher_epochs=25,map_updates=600)
        # Paired teacher-strength control. Same initialization/data recipe, old 60-epoch training.
        Xt,yt,Xv,yv,Xte,yte,data_meta=base.datasets()
        baseline=base.train_model(seed,Xt,yt,input_dim=data_meta['input_dim'],epochs=60)
        baseline_val=base.r2(baseline,Xv,yv);baseline_test=base.r2(baseline,Xte,yte)
        baseline_state=state_sha256(baseline)
    finally:
        restore()
    if counts['float32_sqrt_calls']<=0 or counts['adamw_steps']<=0:
        raise RuntimeError('sqrt64 AdamW intervention did not execute')
    result.update({
        'experiment':'PHASE3B_STRONGER_TEACHER_CONFIRMATORY',
        'evidence_class':'PROSPECTIVE_CONFIRMATORY' if seed in protocol['fresh_model_seeds'] else 'PREFLIGHT_ONLY',
        'stage_a_selected_recipe':'short_25',
        'candidate_teacher_val_r2':result['teacher_val_r2'],
        'candidate_teacher_test_r2':result['teacher_test_r2'],
        'baseline60_teacher_val_r2':baseline_val,
        'baseline60_teacher_test_r2':baseline_test,
        'candidate_minus_baseline_test_r2':result['teacher_test_r2']-baseline_test,
        'baseline60_teacher_state_sha256':baseline_state,
        'numeric_profile':{
            'ATEN_CPU_CAPABILITY':'avx2','effective_cpu_capability':torch.backends.cpu.get_cpu_capability(),
            'MKL_CBWR':os.environ.get('MKL_CBWR'),'torch_threads':torch.get_num_threads(),
            'mkldnn_enabled':torch.backends.mkldnn.enabled,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'sqrt_intervention':counts
        },
        'provenance':{
            'stage_b_protocol_sha256':sha256(PROTOCOL),'base_runner_git_blob_expected':'8dd3ddefb59ba250f5deca2d98d04fbff0055748',
            'wrapper_sha256':sha256(Path(__file__)),'python':platform.python_version(),'torch':torch.__version__,
            'numpy':np.__version__,'scikit_learn':sklearn.__version__,'platform':platform.platform()
        }
    })
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'seed':seed,'candidate_test_r2':result['candidate_teacher_test_r2'],'baseline_test_r2':baseline_test,
                      'sep_budget':None if result['selected_sep'] is None else result['selected_sep']['budget'],
                      'comp_budget':None if result['selected_comp'] is None else result['selected_comp']['budget'],
                      'sqrt_calls':counts['float32_sqrt_calls']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.seed,a.out)
