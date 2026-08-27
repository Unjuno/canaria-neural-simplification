from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recursive_composition.exploration.c4_smallvit_recursive import (
    Cluster,
    ReplacedViT,
    SmallViT,
    TinyTokenRes,
    accuracy,
    collect_span,
    count_params,
    data_split,
    fit_map,
    set_seed,
    set_trainable,
    train_teacher,
)


torch.set_num_threads(1)


def nmse(pred, target, denom=None):
    if denom is None:
        denom = float(((target - target.mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12
    return float(F.mse_loss(pred, target)) / denom


def fixed_basis():
    rng = np.random.default_rng(20261030)
    a = rng.standard_normal((32, 32))
    q, r = np.linalg.qr(a)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    q = q * signs[None, :]
    return torch.tensor(q.astype(np.float32))


def adapt_sketch(cluster, xin, target, projection, seed):
    set_trainable(cluster, [0, 1])
    params = [p for p in cluster.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=8e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    target_proj = target @ projection
    for _ in range(600):
        ix = torch.randint(0, len(xin), (64,), generator=gen)
        opt.zero_grad()
        loss = F.mse_loss(cluster(xin[ix]) @ projection, target_proj[ix])
        loss.backward()
        opt.step()
    set_trainable(cluster, [])
    return cluster


def adapt_full(cluster, xin, target, seed):
    set_trainable(cluster, [0, 1])
    params = [p for p in cluster.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=8e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(600):
        ix = torch.randint(0, len(xin), (64,), generator=gen)
        opt.zero_grad()
        loss = F.mse_loss(cluster(xin[ix]), target[ix])
        loss.backward()
        opt.step()
    set_trainable(cluster, [])
    return cluster


def run(seed: int):
    set_seed(seed)
    tr, va = data_split()
    teacher = train_teacher(SmallViT(), tr, seed + 50000, 45)
    teacher_val = accuracy(teacher, va)
    result = {"seed":seed,"status":"EXPLORATORY_OUTCOME","test_evaluated":False,"teacher_val_acc":teacher_val,"eligible":bool(teacher_val >= 0.95)}
    if not result["eligible"]:
        return result

    a0t, a1t, a2t = collect_span(teacher, tr, start=1)
    a0v, _a1v, a2v = collect_span(teacher, va, start=1)
    a0fit, a1fit, a2fit = a0t[:512], a1t[:512], a2t[:512]
    denom = float(((a2v - a2v.mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12

    c1 = fit_map(TinyTokenRes(32,32,seed+101001), a0fit, a1fit, 600, seed+102001)
    c2 = fit_map(TinyTokenRes(32,32,seed+101002), a1fit, a2fit, 600, seed+102002)
    base = Cluster([c1,c2])
    set_trainable(base, [])
    with torch.no_grad():
        baseline_fit = base(a0fit).detach()

    assert count_params(base) == 4096
    basis = fixed_basis(); p8=basis[:,:8]; p16=basis[:,:16]
    align_seed=seed+120000; final_init=seed+130000; final_fit=seed+140000
    conditions={}

    for name in ("frozen","sketch_only_16","anchored_8","anchored_16","full_32"):
        cluster=copy.deepcopy(base)
        if name == "sketch_only_16":
            adapt_sketch(cluster,a0fit,a2fit,p16,align_seed)
        elif name in ("anchored_8","anchored_16"):
            p=p8 if name=="anchored_8" else p16
            with torch.no_grad():
                correction=((a2fit-baseline_fit)@p)@p.T
                target=(baseline_fit+correction).detach()
            adapt_full(cluster,a0fit,target,align_seed)
        elif name == "full_32":
            adapt_full(cluster,a0fit,a2fit,align_seed)
        set_trainable(cluster, [])

        with torch.no_grad():
            cluster_fit=cluster(a0fit).detach(); cluster_val=cluster(a0v).detach()
        cluster_denom=float(((cluster_val-cluster_val.mean(dim=(0,1),keepdim=True))**2).mean())+1e-12
        final=fit_map(TinyTokenRes(32,64,final_init),a0fit,cluster_fit,600,final_fit)
        with torch.no_grad(): final_val=final(a0v)
        conditions[name]={
            "cluster_nmse_vs_original":nmse(cluster_val,a2v,denom),
            "cluster_replacement_val_acc":accuracy(ReplacedViT(teacher,copy.deepcopy(cluster),start=1),va),
            "final_nmse_vs_cluster":nmse(final_val,cluster_val,cluster_denom),
            "final_nmse_vs_original":nmse(final_val,a2v,denom),
            "final_replacement_val_acc":accuracy(ReplacedViT(teacher,copy.deepcopy(final),start=1),va),
        }

    direct=fit_map(TinyTokenRes(32,64,final_init),a0fit,a2fit,600,final_fit)
    assert count_params(direct)==4096
    with torch.no_grad(): direct_val=direct(a0v)
    direct_nmse=nmse(direct_val,a2v,denom); direct_acc=accuracy(ReplacedViT(teacher,copy.deepcopy(direct),start=1),va)
    frozen_nmse=conditions["frozen"]["final_nmse_vs_original"]; full_nmse=conditions["full_32"]["final_nmse_vs_original"]
    for rec in conditions.values():
        rec["difference_final_vs_frozen"]=rec["final_nmse_vs_original"]-frozen_nmse
        rec["ratio_final_over_full32"]=rec["final_nmse_vs_original"]/full_nmse
        rec["ratio_final_over_direct"]=rec["final_nmse_vs_original"]/direct_nmse

    result.update({
        "exact_parameter_match":count_params(base)==count_params(direct)==4096,
        "basis":{"rng_seed":20261030,"nested_dimensions":[8,16]},
        "conditions":conditions,
        "direct_original_single":{"final_nmse_vs_original":direct_nmse,"replacement_val_acc":direct_acc},
        "ordering_final_nmse_best_to_worst":sorted(conditions,key=lambda n:conditions[n]["final_nmse_vs_original"]),
        "ordering_final_accuracy_best_to_worst":sorted(conditions,key=lambda n:-conditions[n]["final_replacement_val_acc"]),
    })
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
    rec=run(args.seed); args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(rec,indent=2),encoding="utf-8"); print(json.dumps(rec,indent=2))

if __name__ == "__main__": main()
