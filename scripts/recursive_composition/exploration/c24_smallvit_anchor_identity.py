from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.recursive_composition.exploration.c4_smallvit_recursive import Cluster,ReplacedViT,SmallViT,TinyTokenRes,accuracy,collect_span,count_params,data_split,fit_map,set_seed,set_trainable,train_teacher

torch.set_num_threads(1)

def nmse(pred,target,denom=None):
    if denom is None: denom=float(((target-target.mean(dim=(0,1),keepdim=True))**2).mean())+1e-12
    return float(F.mse_loss(pred,target))/denom

def projection():
    rng=np.random.default_rng(20261310); a=rng.standard_normal((32,32)); q,r=np.linalg.qr(a); signs=np.where(np.diag(r)<0,-1.0,1.0); q=q*signs[None,:]
    return torch.tensor(q[:,:8].astype(np.float32))

def adapt_sketch(cluster,xin,target,p,seed):
    set_trainable(cluster,[0,1]); opt=torch.optim.AdamW([x for x in cluster.parameters() if x.requires_grad],lr=8e-3,weight_decay=1e-5); gen=torch.Generator().manual_seed(seed); tp=target@p
    for _ in range(600):
        ix=torch.randint(0,len(xin),(64,),generator=gen); opt.zero_grad(); loss=F.mse_loss(cluster(xin[ix])@p,tp[ix]); loss.backward(); opt.step()
    set_trainable(cluster,[]); return cluster

def adapt_full(cluster,xin,target,seed):
    set_trainable(cluster,[0,1]); opt=torch.optim.AdamW([x for x in cluster.parameters() if x.requires_grad],lr=8e-3,weight_decay=1e-5); gen=torch.Generator().manual_seed(seed)
    for _ in range(600):
        ix=torch.randint(0,len(xin),(64,),generator=gen); opt.zero_grad(); loss=F.mse_loss(cluster(xin[ix]),target[ix]); loss.backward(); opt.step()
    set_trainable(cluster,[]); return cluster

def compile_final(teacher,cluster,a0fit,a0v,a2v,denom,va,init_seed,fit_seed):
    set_trainable(cluster,[])
    with torch.no_grad(): cf=cluster(a0fit).detach(); cv=cluster(a0v).detach()
    cd=float(((cv-cv.mean(dim=(0,1),keepdim=True))**2).mean())+1e-12
    final=fit_map(TinyTokenRes(32,64,init_seed),a0fit,cf,600,fit_seed)
    with torch.no_grad(): fv=final(a0v)
    return final,{"cluster_nmse_vs_original":nmse(cv,a2v,denom),"final_nmse_vs_cluster":nmse(fv,cv,cd),"final_nmse_vs_original":nmse(fv,a2v,denom),"cluster_replacement_val_acc":accuracy(ReplacedViT(teacher,copy.deepcopy(cluster),start=1),va),"final_replacement_val_acc":accuracy(ReplacedViT(teacher,copy.deepcopy(final),start=1),va)}

def run(seed):
    set_seed(seed); tr,va=data_split(); teacher=train_teacher(SmallViT(),tr,seed+50000,45); tacc=accuracy(teacher,va)
    rec={"seed":seed,"status":"EXPLORATORY_OUTCOME","test_evaluated":False,"teacher_val_acc":tacc,"eligible":bool(tacc>=0.95)}
    if not rec["eligible"]: return rec
    a0t,a1t,a2t=collect_span(teacher,tr,start=1); a0v,_a1v,a2v=collect_span(teacher,va,start=1); a0fit,a1fit,a2fit=a0t[:512],a1t[:512],a2t[:512]; denom=float(((a2v-a2v.mean(dim=(0,1),keepdim=True))**2).mean())+1e-12
    c1=fit_map(TinyTokenRes(32,32,seed+101001),a0fit,a1fit,600,seed+102001); c2=fit_map(TinyTokenRes(32,32,seed+101002),a1fit,a2fit,600,seed+102002); base=Cluster([c1,c2]); set_trainable(base,[])
    with torch.no_grad(): baseline=base(a0fit).detach()
    p=projection(); mean=baseline.mean(dim=0,keepdim=True).expand_as(baseline).detach(); gen=torch.Generator().manual_seed(seed+180000); shuffled=baseline[torch.randperm(len(baseline),generator=gen)].detach()
    anchors={"anchor_self":baseline,"anchor_input":a0fit.detach(),"anchor_mean":mean,"anchor_shuffled":shuffled,"anchor_zero":torch.zeros_like(baseline)}
    align_seed=seed+120000; init=seed+130000; fseed=seed+140000; conditions={}
    for name in ["frozen","sketch_only_8","anchor_self","anchor_input","anchor_mean","anchor_shuffled","anchor_zero","full_32"]:
        cl=copy.deepcopy(base)
        if name=="sketch_only_8": adapt_sketch(cl,a0fit,a2fit,p,align_seed)
        elif name in anchors:
            A=anchors[name]
            with torch.no_grad(): target=(A+((a2fit-A)@p)@p.T).detach()
            adapt_full(cl,a0fit,target,align_seed)
        elif name=="full_32": adapt_full(cl,a0fit,a2fit,align_seed)
        final,m=compile_final(teacher,cl,a0fit,a0v,a2v,denom,va,init,fseed); conditions[name]=m
    direct=fit_map(TinyTokenRes(32,64,init),a0fit,a2fit,600,fseed)
    with torch.no_grad(): dv=direct(a0v)
    dnm=nmse(dv,a2v,denom); dacc=accuracy(ReplacedViT(teacher,copy.deepcopy(direct),start=1),va); full=conditions["full_32"]["final_nmse_vs_original"]; fro=conditions["frozen"]["final_nmse_vs_original"]
    for m in conditions.values(): m["ratio_final_over_direct"]=m["final_nmse_vs_original"]/dnm; m["ratio_final_over_full32"]=m["final_nmse_vs_original"]/full; m["difference_final_vs_frozen"]=m["final_nmse_vs_original"]-fro
    rec.update({"exact_parameter_match":count_params(base)==count_params(direct)==4096,"basis":{"rng_seed":20261310,"dimension":8},"conditions":conditions,"direct_original_single":{"final_nmse_vs_original":dnm,"replacement_val_acc":dacc},"ordering_final_nmse_best_to_worst":sorted(conditions,key=lambda k:conditions[k]["final_nmse_vs_original"])})
    return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args(); r=run(a.seed); a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
if __name__=='__main__': main()
