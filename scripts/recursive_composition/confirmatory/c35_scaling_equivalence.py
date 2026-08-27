from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.recursive_composition.exploration.c4_smallvit_recursive import (
    Cluster, ReplacedViT, SmallViT, TinyTokenRes, accuracy, collect_span,
    count_params, fit_map, set_seed, set_trainable, train_teacher,
)

torch.set_num_threads(1)
POOL_COUNTS = {512:[128,256,384,512], 1024:[128,256,384,512,768,1024]}

def nmse(pred,target,denom=None):
    if denom is None:
        denom=float(((target-target.mean(dim=(0,1),keepdim=True))**2).mean())+1e-12
    return float(F.mse_loss(pred,target))/denom

def split_arrays_no_test_materialization():
    d=load_digits(); X=(d.images.astype(np.float32)/16.0)[:,None,:,:]; y=d.target.astype(np.int64); idx=np.arange(len(y))
    tr,temp=train_test_split(idx,test_size=0.30,random_state=24680,stratify=y)
    va,_te=train_test_split(temp,test_size=0.50,random_state=13579,stratify=y[temp])
    return X,y,tr,va

def ds(X,y,ii): return torch.utils.data.TensorDataset(torch.from_numpy(X[ii]),torch.from_numpy(y[ii]))

def projection():
    rng=np.random.default_rng(20261320); a=rng.standard_normal((32,32)); q,r=np.linalg.qr(a); signs=np.where(np.diag(r)<0,-1.0,1.0); q=q*signs[None,:]
    return torch.tensor(q[:,:8].astype(np.float32))

def adapt_full(cluster,xin,target,seed):
    set_trainable(cluster,[0,1]); opt=torch.optim.AdamW([p for p in cluster.parameters() if p.requires_grad],lr=8e-3,weight_decay=1e-5); gen=torch.Generator().manual_seed(seed)
    for _ in range(600):
        ix=torch.randint(0,len(xin),(64,),generator=gen); opt.zero_grad(); loss=F.mse_loss(cluster(xin[ix]),target[ix]); loss.backward(); opt.step()
    set_trainable(cluster,[]); return cluster

def compile_final(teacher,cluster,a0fit,a0v,a2v,denom,va,init_seed,fit_seed):
    set_trainable(cluster,[])
    with torch.no_grad(): cf=cluster(a0fit).detach(); cv=cluster(a0v).detach()
    cd=float(((cv-cv.mean(dim=(0,1),keepdim=True))**2).mean())+1e-12
    final=fit_map(TinyTokenRes(32,64,init_seed),a0fit,cf,600,fit_seed)
    with torch.no_grad(): fv=final(a0v)
    return {"cluster_nmse_vs_original":nmse(cv,a2v,denom),"final_nmse_vs_cluster":nmse(fv,cv,cd),"final_nmse_vs_original":nmse(fv,a2v,denom),"cluster_replacement_val_acc":accuracy(ReplacedViT(teacher,copy.deepcopy(cluster),start=1),va),"final_replacement_val_acc":accuracy(ReplacedViT(teacher,copy.deepcopy(final),start=1),va)}

def run_pool(seed,teacher,va,a0t,a1t,a2t,a0v,a2v,pool_size):
    a0fit,a1fit,a2fit=a0t[:pool_size],a1t[:pool_size],a2t[:pool_size]
    denom=float(((a2v-a2v.mean(dim=(0,1),keepdim=True))**2).mean())+1e-12
    c1=fit_map(TinyTokenRes(32,32,seed+101001),a0fit,a1fit,600,seed+102001)
    c2=fit_map(TinyTokenRes(32,32,seed+101002),a1fit,a2fit,600,seed+102002)
    base=Cluster([c1,c2]); set_trainable(base,[]); assert count_params(base)==4096
    with torch.no_grad(): baseline=base(a0fit).detach()
    p=projection(); order_seed=seed+190000+pool_size; order=torch.randperm(pool_size,generator=torch.Generator().manual_seed(order_seed))
    align_seed=seed+120000; init_seed=seed+130000; final_fit_seed=seed+140000; conditions={}
    conditions['frozen']=compile_final(teacher,copy.deepcopy(base),a0fit,a0v,a2v,denom,va,init_seed,final_fit_seed)
    for k in POOL_COUNTS[pool_size]:
        ix=order[:k]
        with torch.no_grad(): target=(baseline[ix]+((a2fit[ix]-baseline[ix])@p)@p.T).detach()
        cl=copy.deepcopy(base); adapt_full(cl,a0fit[ix],target,align_seed)
        m=compile_final(teacher,cl,a0fit,a0v,a2v,denom,va,init_seed,final_fit_seed); m['calibration_unique_samples']=int(k); m['calibration_fraction_of_fit_pool']=float(k/pool_size); conditions[f'self_{k}']=m
    full=copy.deepcopy(base); adapt_full(full,a0fit,a2fit,align_seed); conditions['full_32']=compile_final(teacher,full,a0fit,a0v,a2v,denom,va,init_seed,final_fit_seed)
    direct=fit_map(TinyTokenRes(32,64,init_seed),a0fit,a2fit,600,final_fit_seed); assert count_params(direct)==4096
    with torch.no_grad(): dv=direct(a0v)
    dnm=nmse(dv,a2v,denom); dacc=accuracy(ReplacedViT(teacher,copy.deepcopy(direct),start=1),va)
    frozen_nm=conditions['frozen']['final_nmse_vs_original']; full_nm=conditions['full_32']['final_nmse_vs_original']; recovery_denom=frozen_nm-full_nm
    for m in conditions.values():
        m['difference_final_vs_frozen']=m['final_nmse_vs_original']-frozen_nm; m['ratio_final_over_full32']=m['final_nmse_vs_original']/full_nm; m['ratio_final_over_direct']=m['final_nmse_vs_original']/dnm; m['validation_accuracy_difference_vs_full32']=m['final_replacement_val_acc']-conditions['full_32']['final_replacement_val_acc']; m['recovery_fraction_toward_full32']=((frozen_nm-m['final_nmse_vs_original'])/recovery_denom) if recovery_denom>0 else None
    return {'fit_pool_samples':pool_size,'calibration_counts':POOL_COUNTS[pool_size],'calibration_order_seed':order_seed,'exact_parameter_match':count_params(base)==count_params(direct)==4096,'recovery_denominator_positive':bool(recovery_denom>0),'conditions':conditions,'direct_original_single':{'final_nmse_vs_original':dnm,'replacement_val_acc':dacc}}

def run(seed):
    set_seed(seed); X,y,tr_i,va_i=split_arrays_no_test_materialization(); tr=ds(X,y,tr_i); va=ds(X,y,va_i); teacher=train_teacher(SmallViT(),tr,seed+50000,45); tacc=accuracy(teacher,va)
    rec={'seed':seed,'status':'CONFIRMATORY_SEED_OUTCOME','teacher_val_acc':tacc,'eligible':bool(tacc>=0.95),'test_evaluated':False}
    if not rec['eligible']: return rec
    a0t,a1t,a2t=collect_span(teacher,tr,start=1); a0v,_a1v,a2v=collect_span(teacher,va,start=1); assert len(a0t)>=1024
    pools={str(ps):run_pool(seed,teacher,va,a0t,a1t,a2t,a0v,a2v,ps) for ps in [512,1024]}
    scaling={'valid_recovery_denominators':all(pools[str(ps)]['recovery_denominator_positive'] for ps in [512,1024])}
    if scaling['valid_recovery_denominators']:
        r512={k:pools['512']['conditions'][f'self_{k}']['recovery_fraction_toward_full32'] for k in [256,384,512]}
        r1024_abs={k:pools['1024']['conditions'][f'self_{k}']['recovery_fraction_toward_full32'] for k in [256,384,512]}
        abs_errors=[abs(r512[k]-r1024_abs[k]) for k in [256,384,512]]
        frac_pairs=[(256,512),(384,768),(512,1024)]
        frac_errors=[abs(pools['512']['conditions'][f'self_{a}']['recovery_fraction_toward_full32']-pools['1024']['conditions'][f'self_{b}']['recovery_fraction_toward_full32']) for a,b in frac_pairs]
        scaling.update({'mean_abs_matched_recovery_error':float(np.mean(abs_errors)),'mean_fraction_matched_recovery_error':float(np.mean(frac_errors)),'error_difference_abs_minus_fraction':float(np.mean(abs_errors)-np.mean(frac_errors)),'abs_errors_by_count':{str(k):float(e) for k,e in zip([256,384,512],abs_errors)},'fraction_errors_by_fraction':{'0.50':float(frac_errors[0]),'0.75':float(frac_errors[1]),'1.00':float(frac_errors[2])}})
    rec.update({'basis':{'rng_seed':20261320,'dimension':8,'ambient_dimension':32},'pools':pools,'scaling_contrast':scaling}); return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); r=run(a.seed); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2),encoding='utf-8'); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
