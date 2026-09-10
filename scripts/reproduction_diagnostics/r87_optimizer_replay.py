#!/usr/bin/env python3
"""Post-hoc common-input replay. sqrt64 is a separately labelled intervention."""
from __future__ import annotations
import argparse,hashlib,importlib,importlib.util,inspect,json,math,os,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(trace,seed,out,capability,sqrt64=False):
    os.environ.update(ATEN_CPU_CAPABILITY=capability,MKL_CBWR='COMPATIBLE',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_DYNAMIC='FALSE',OMP_DYNAMIC='FALSE')
    import numpy as np,torch
    adam_module=importlib.import_module('torch.optim.adam');adam_source=inspect.getfile(adam_module)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.mkldnn.enabled=False;torch.use_deterministic_algorithms(True)
    original_sqrt=torch.Tensor.sqrt
    if sqrt64:
        def guarded_sqrt(t):
            return original_sqrt(t.to(torch.float64)).to(t.dtype) if t.dtype==torch.float32 else original_sqrt(t)
        torch.Tensor.sqrt=guarded_sqrt
    spec=importlib.util.spec_from_file_location('science',ROOT/'scripts/reproduce/core_discovery_digits/run_confirmatory.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    x=np.load(trace,allow_pickle=False);pflat=x['initial_parameters'];gflat=x['first_gradients'];model=r.Net(seed);offset=0;stages={};names={}
    for name,p in model.named_parameters():
        n=p.numel();p.data.copy_(torch.from_numpy(pflat[offset:offset+n].copy()).reshape(p.shape));p.grad=torch.from_numpy(gflat[offset:offset+n].copy()).reshape(p.shape);names[name]=list(p.shape);offset+=n
    assert offset==len(pflat)==len(gflat)
    def flatten(xs):return torch.cat([z.detach().reshape(-1) for z in xs]).numpy().copy()
    manual=[p.detach().clone() for p in model.parameters()];grads=[p.grad.clone() for p in model.parameters()]
    m=[torch.zeros_like(p) for p in manual];v=[torch.zeros_like(p) for p in manual]
    with torch.no_grad():
        for p in manual:p.mul_(1-.002*.0001)
        stages['weight_decay']=flatten(manual)
        for a,g in zip(m,grads):a.lerp_(g,1-.9)
        stages['first_moment']=flatten(m)
        for a,g in zip(v,grads):a.mul_(.999).addcmul_(g,g,value=1-.999)
        stages['second_moment']=flatten(v)
        sq=[a.sqrt() for a in v];stages['square_root']=flatten(sq)
        bc1=1-.9**1.;bc2=1-.999**1.;bc2sqrt=bc2**.5;step=.002/bc1
        denom=[a/bc2sqrt for a in sq];stages['normalized_root']=flatten(denom)
        for a in denom:a.add_(1e-8)
        stages['denominator']=flatten(denom)
        for p,a,b in zip(manual,m,denom):p.addcdiv_(a,b,value=-step)
        stages['manual_updated_parameters']=flatten(manual)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001);opt.step()
    stages['actual_updated_parameters']=flatten(list(model.parameters()))
    stages['actual_first_moment']=flatten([opt.state[p]['exp_avg'] for p in model.parameters()]);stages['actual_second_moment']=flatten([opt.state[p]['exp_avg_sq'] for p in model.parameters()])
    torch.Tensor.sqrt=original_sqrt
    out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'optimizer_arrays.npz',**stages)
    build=Path(torch.__file__).parent/'lib/libtorch_cpu.so'
    report={'experiment':'R87D6_COMMON_INPUT_OPTIMIZER_REPLAY','evidence_class':'POST_HOC_OPERATOR_REPLAY','seed':seed,'sqrt64':sqrt64,'source_sha256':sha(__file__),'input_trace_sha256':sha(trace),'array_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in stages.items()},'manual_matches_actual':bool(np.array_equal(stages['manual_updated_parameters'],stages['actual_updated_parameters'])),'matches_saved_original_step1':bool(np.array_equal(stages['actual_updated_parameters'],x['parameters_step_1'])),'scalars_hex':{k:float(v).hex() for k,v in {'beta1':.9,'beta2':.999,'bc1':bc1,'bc2':bc2,'bc2sqrt_pow':bc2sqrt,'bc2sqrt_sqrt':math.sqrt(bc2),'step_size':step}.items()},'torch':torch.__version__,'torch_cpu_library_sha256':sha(build),'adam_python_sha256':sha(adam_source),'python':sys.version,'platform':platform.platform(),'libc':platform.libc_ver(),'cpu':subprocess.check_output(['lscpu'],text=True),'effective_capability':torch.backends.cpu.get_cpu_capability(),'parameter_shapes':names}
    (out/'REPORT.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--trace',type=Path,required=True);a.add_argument('--seed',type=int,default=1200);a.add_argument('--out',type=Path,required=True);a.add_argument('--capability',choices=('avx2','default'),default='avx2');a.add_argument('--sqrt64',action='store_true');x=a.parse_args();run(x.trace,x.seed,x.out,x.capability,x.sqrt64)
