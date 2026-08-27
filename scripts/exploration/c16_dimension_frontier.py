from __future__ import annotations
import argparse, copy, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from scripts.exploration.c10_boundary_signal_ablation import (
    Chain, FullSpanReplacedNet, TinyRes, accuracy, acts,
    compile_final_from_hierarchy, count_params, fit_map,
    set_all_trainable, split_data, train_teacher,
)
from scripts.exploration.c12_self_anchored_sketches import adapt_anchored
from scripts.exploration.c14_basis_robustness import random_orthogonal_basis

DIMS=[8,16,24,32]

def basis_family():
    eye=torch.eye(64,dtype=torch.float32)
    return {
        "identity":eye,
        "random_20261001":random_orthogonal_basis(20261001),
        "random_20261002":random_orthogonal_basis(20261002),
    }

def run(seed:int):
    Xt,yt,Xv,yv=split_data()
    teacher=train_teacher(seed,Xt,yt,60)
    at=acts(teacher,Xt); av=acts(teacher,Xv)
    a0t,a1t,a2t,a3t,a4t=at
    a0v,_a1v,a2v,_a3v,a4v=av
    denom=float(((a4v-a4v.mean(0,keepdim=True))**2).mean())+1e-12

    locals4=[
        fit_map(TinyRes(64,8,seed+101001),a0t,a1t,600,seed+102001),
        fit_map(TinyRes(64,8,seed+101002),a1t,a2t,600,seed+102002),
        fit_map(TinyRes(64,8,seed+101003),a2t,a3t,600,seed+102003),
        fit_map(TinyRes(64,8,seed+101004),a3t,a4t,600,seed+102004),
    ]
    pair12=Chain([copy.deepcopy(locals4[0]),copy.deepcopy(locals4[1])])
    set_all_trainable(pair12,True); fit_map(pair12,a0t,a2t,600,seed+110001); set_all_trainable(pair12,False)
    pair34=Chain([copy.deepcopy(locals4[2]),copy.deepcopy(locals4[3])])
    set_all_trainable(pair34,True); fit_map(pair34,a2t,a4t,600,seed+110002); set_all_trainable(pair34,False)
    with torch.no_grad():
        p12t=pair12(a0t).detach(); p34t=pair34(a2t).detach()
    c12=fit_map(TinyRes(64,16,seed+120001),a0t,p12t,600,seed+121001)
    c34=fit_map(TinyRes(64,16,seed+120002),a2t,p34t,600,seed+121002)
    base=Chain([copy.deepcopy(c12),copy.deepcopy(c34)])
    set_all_trainable(base,False)
    with torch.no_grad(): baseline_t=base(a0t).detach()

    top_seed=seed+150000; final_init=seed+160000; final_fit=seed+170000
    conditions={}

    frozen=copy.deepcopy(base)
    ffinal,fm=compile_final_from_hierarchy(frozen,a0t,a0v,a4v,denom,final_init,final_fit)
    fm["final_replacement_val_acc"]=accuracy(FullSpanReplacedNet(teacher,copy.deepcopy(ffinal)),Xv,yv)
    conditions["frozen"]=fm

    for bname,basis in basis_family().items():
        for k in DIMS:
            name=f"{bname}_k{k}"
            p=basis[:,:k]
            h=copy.deepcopy(base)
            with torch.no_grad():
                correction=((a4t-baseline_t)@p)@p.T
                target=(baseline_t+correction).detach()
            adapt_anchored(h,a0t,target,600,top_seed)
            set_all_trainable(h,False)
            final,m=compile_final_from_hierarchy(h,a0t,a0v,a4v,denom,final_init,final_fit)
            m["final_replacement_val_acc"]=accuracy(FullSpanReplacedNet(teacher,copy.deepcopy(final)),Xv,yv)
            conditions[name]=m

    full=copy.deepcopy(base)
    adapt_anchored(full,a0t,a4t.detach(),600,top_seed)
    set_all_trainable(full,False)
    full_final,fullm=compile_final_from_hierarchy(full,a0t,a0v,a4v,denom,final_init,final_fit)
    fullm["final_replacement_val_acc"]=accuracy(FullSpanReplacedNet(teacher,copy.deepcopy(full_final)),Xv,yv)
    conditions["full_64"]=fullm

    direct=TinyRes(64,32,final_init)
    direct=fit_map(direct,a0t,a4t,600,final_fit)
    with torch.no_grad(): direct_v=direct(a0v)
    direct_nmse=float(torch.nn.functional.mse_loss(direct_v,a4v))/denom
    direct_acc=accuracy(FullSpanReplacedNet(teacher,copy.deepcopy(direct)),Xv,yv)

    frozen_nmse=conditions["frozen"]["final_nmse_vs_original"]
    full_nmse=conditions["full_64"]["final_nmse_vs_original"]
    dimension_summary={}
    for k in DIMS:
        names=[f"{b}_k{k}" for b in basis_family()]
        for n in names:
            conditions[n]["difference_vs_frozen"]=conditions[n]["final_nmse_vs_original"]-frozen_nmse
            conditions[n]["ratio_over_full64"]=conditions[n]["final_nmse_vs_original"]/full_nmse
        vals=[conditions[n]["final_nmse_vs_original"] for n in names]
        dimension_summary[str(k)]={
            "worst_nmse":max(vals),
            "best_nmse":min(vals),
            "worst_over_full64":max(vals)/full_nmse,
            "worst_minus_frozen":max(vals)-frozen_nmse,
            "all_bases_improve_frozen":all(conditions[n]["difference_vs_frozen"]<0 for n in names),
        }

    return {
        "seed":seed,
        "status":"EXPLORATORY_OUTCOME",
        "test_evaluated":False,
        "teacher_val_acc":accuracy(teacher,Xv,yv),
        "budget":{"local_total_params":sum(count_params(m) for m in locals4),"level1_total_params":count_params(c12)+count_params(c34),"final_params":count_params(direct),"exact_4096_each_level":sum(count_params(m) for m in locals4)==count_params(c12)+count_params(c34)==count_params(direct)==4096},
        "conditions":conditions,
        "dimension_summary":dimension_summary,
        "direct_original_single":{"final_nmse_vs_original":direct_nmse,"replacement_val_acc":direct_acc},
    }

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
    rec=run(args.seed); args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(rec,indent=2),encoding="utf-8"); print(json.dumps(rec,indent=2))
if __name__=="__main__": main()
