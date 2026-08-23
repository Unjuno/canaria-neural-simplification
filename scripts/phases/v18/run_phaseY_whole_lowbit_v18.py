import os,argparse,importlib.util,math,json
from copy import deepcopy
import numpy as np,pandas as pd,torch
SRC='/mnt/data/canaria_v18_global_accounting/scripts/run_phaseX_global_accounting_v18.py'
spec=importlib.util.spec_from_file_location('X',SRC); X=importlib.util.module_from_spec(spec); spec.loader.exec_module(X)
OUT='/mnt/data/canaria_v18_global_accounting/raw_phaseY';os.makedirs(OUT,exist_ok=True)
BITS=[3,4,6,8]; ALPHAS=[.50,.65,.80,.90,1.00,1.10,1.25]

def qparam(p,bits):
    x=p.detach().cpu().float(); qmax=2**(bits-1)-1; mx=float(x.abs().max()); base=mx/qmax if mx>1e-12 else 1.;best=None
    for a in ALPHAS:
        sc=float(np.float32(np.float16(base*a))); q=torch.clamp(torch.round(x/sc),-qmax,qmax); xr=q*sc; mse=float(((xr-x)**2).mean())
        if best is None or mse<best[0]:best=(mse,sc,xr)
    return best[2]
def qmodel(m,bits,segs):
    mm=deepcopy(m); nt=0; ns=0
    with torch.no_grad():
        for seg in segs:
            for _,p in getattr(mm,seg).named_parameters(): p.copy_(qparam(p,bits));nt+=p.numel();ns+=1
    return mm,nt,ns

def run(seed):
    od=os.path.join(OUT,f'seed_{seed}');os.makedirs(od,exist_ok=True)
    base,data=X.train_base(seed);Xtr,ytr,Xv,yv=data;Xa=X.make_aug(Xv,123);bc=X.acc(base,Xv,yv)
    if bc<.95: print('INELIGIBLE',seed);return
    with torch.no_grad(): Z=X.span_input(base,Xtr[:X.CALN]);Y=X.span_output(base,Z)
    A=X.design(Z);B=X.targets(Y);ref=X.fit_full(Z,Y);W=X.core_to_matrix(ref);mask,pbits=X.shared_nm1_mask(W,8);Wr=X.refit_mask(A,B,mask);Q=X.quant_group_fp16(Wr,A,B,1,2);core=X.matrix_to_core(Q)
    RX,RY=X.repair_data(data,seed)
    control=deepcopy(base)
    for p in control.parameters():p.requires_grad_(True)
    control=X.train8(control,RX,RY,seed+70000)
    compiled=X.set_full_shell(X.replace_full(base,core));compiled=X.train8(compiled,RX,RY,seed+70000)
    ca=X.acc(control,Xa,yv);xa=X.acc(compiled,Xa,yv)
    rows=[]
    for b in BITS:
        qc,nc,tc=qmodel(control,b,X.SEGMENTS); qx,nx,tx=qmodel(compiled,b,('stem','b_in','b_out','head'))
        ac=X.acc(qc,Xa,yv); ax=X.acc(qx,Xa,yv)
        cbits=nc*b+tc*16; xbits=nx*b+tx*16+X.CORE_BITS
        rows.append(dict(seed=seed,bits=b,control_fp_aug=ca,compiled_fp_aug=xa,control_q_aug=ac,compiled_q_aug=ax,control_fidelity=ac/(ca+1e-12),compiled_fidelity=ax/(xa+1e-12),quantized_utility=ax/(ac+1e-12),control_packed_bytes=cbits/8,compiled_packed_bytes=xbits/8,reduction=1-xbits/cbits))
    pd.DataFrame(rows).to_csv(os.path.join(od,'results.csv'),index=False);print('DONE',seed,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();run(a.seed)
