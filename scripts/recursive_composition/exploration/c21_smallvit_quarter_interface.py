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

BASIS_SPECS = [
    ("identity_first8", None),
    ("random_20261120", 20261120),
    ("random_20261121", 20261121),
    ("random_20261122", 20261122),
]


def nmse(pred, target, denom=None):
    if denom is None:
        denom = float(((target - target.mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12
    return float(F.mse_loss(pred, target)) / denom


def basis_projection(seed):
    if seed is None:
        return torch.eye(32, dtype=torch.float32)[:, :8]
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((32, 32))
    q, r = np.linalg.qr(a)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    q = q * signs[None, :]
    return torch.tensor(q[:, :8].astype(np.float32))


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


def compile_final(teacher, cluster, a0fit, a0v, a2v, denom, va, init_seed, fit_seed):
    set_trainable(cluster, [])
    with torch.no_grad():
        cluster_fit = cluster(a0fit).detach()
        cluster_val = cluster(a0v).detach()
    cluster_denom = float(((cluster_val - cluster_val.mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12
    final = fit_map(TinyTokenRes(32, 64, init_seed), a0fit, cluster_fit, 600, fit_seed)
    with torch.no_grad():
        final_val = final(a0v)
    return {
        "cluster_nmse_vs_original": nmse(cluster_val, a2v, denom),
        "cluster_replacement_val_acc": accuracy(ReplacedViT(teacher, copy.deepcopy(cluster), start=1), va),
        "final_nmse_vs_cluster": nmse(final_val, cluster_val, cluster_denom),
        "final_nmse_vs_original": nmse(final_val, a2v, denom),
        "final_replacement_val_acc": accuracy(ReplacedViT(teacher, copy.deepcopy(final), start=1), va),
    }


def run(seed: int):
    set_seed(seed)
    tr, va = data_split()
    teacher = train_teacher(SmallViT(), tr, seed + 50000, 45)
    teacher_val = accuracy(teacher, va)
    result = {
        "seed": seed,
        "status": "EXPLORATORY_OUTCOME",
        "teacher_val_acc": teacher_val,
        "eligible": bool(teacher_val >= 0.95),
        "test_evaluated": False,
    }
    if not result["eligible"]:
        return result

    a0t, a1t, a2t = collect_span(teacher, tr, start=1)
    a0v, _a1v, a2v = collect_span(teacher, va, start=1)
    a0fit, a1fit, a2fit = a0t[:512], a1t[:512], a2t[:512]
    denom = float(((a2v - a2v.mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12

    c1 = fit_map(TinyTokenRes(32, 32, seed + 101001), a0fit, a1fit, 600, seed + 102001)
    c2 = fit_map(TinyTokenRes(32, 32, seed + 101002), a1fit, a2fit, 600, seed + 102002)
    base = Cluster([c1, c2])
    set_trainable(base, [])
    with torch.no_grad():
        baseline_fit = base(a0fit).detach()
    assert count_params(base) == 4096

    align_seed = seed + 120000
    final_init = seed + 130000
    final_fit = seed + 140000

    frozen = compile_final(teacher, copy.deepcopy(base), a0fit, a0v, a2v, denom, va, final_init, final_fit)

    full_cluster = copy.deepcopy(base)
    adapt_full(full_cluster, a0fit, a2fit, align_seed)
    full = compile_final(teacher, full_cluster, a0fit, a0v, a2v, denom, va, final_init, final_fit)

    basis_records = {}
    for basis_name, basis_seed in BASIS_SPECS:
        p8 = basis_projection(basis_seed)

        sketch_cluster = copy.deepcopy(base)
        adapt_sketch(sketch_cluster, a0fit, a2fit, p8, align_seed)
        sketch = compile_final(teacher, sketch_cluster, a0fit, a0v, a2v, denom, va, final_init, final_fit)

        anchored_cluster = copy.deepcopy(base)
        with torch.no_grad():
            correction = ((a2fit - baseline_fit) @ p8) @ p8.T
            anchored_target = (baseline_fit + correction).detach()
        adapt_full(anchored_cluster, a0fit, anchored_target, align_seed)
        anchored = compile_final(teacher, anchored_cluster, a0fit, a0v, a2v, denom, va, final_init, final_fit)

        basis_records[basis_name] = {
            "basis_seed": basis_seed,
            "sketch_only_8": sketch,
            "anchored_8": anchored,
            "anchored_minus_sketch_final_nmse": anchored["final_nmse_vs_original"] - sketch["final_nmse_vs_original"],
            "anchored_minus_frozen_final_nmse": anchored["final_nmse_vs_original"] - frozen["final_nmse_vs_original"],
            "anchored_over_full32": anchored["final_nmse_vs_original"] / full["final_nmse_vs_original"],
            "anchored_val_acc_minus_full32": anchored["final_replacement_val_acc"] - full["final_replacement_val_acc"],
        }

    direct = fit_map(TinyTokenRes(32, 64, final_init), a0fit, a2fit, 600, final_fit)
    assert count_params(direct) == 4096
    with torch.no_grad():
        direct_val = direct(a0v)
    direct_metrics = {
        "final_nmse_vs_original": nmse(direct_val, a2v, denom),
        "replacement_val_acc": accuracy(ReplacedViT(teacher, copy.deepcopy(direct), start=1), va),
    }

    anchored_nmse = {k: v["anchored_8"]["final_nmse_vs_original"] for k, v in basis_records.items()}
    worst_basis = max(anchored_nmse, key=anchored_nmse.get)
    best_basis = min(anchored_nmse, key=anchored_nmse.get)

    result.update({
        "exact_parameter_match": count_params(base) == count_params(direct) == 4096,
        "controls": {"frozen": frozen, "full_32": full, "direct": direct_metrics},
        "bases": basis_records,
        "aggregate": {
            "worst_nmse_basis": worst_basis,
            "best_nmse_basis": best_basis,
            "D_worst": anchored_nmse[worst_basis] - frozen["final_nmse_vs_original"],
            "R_worst": anchored_nmse[worst_basis] / full["final_nmse_vs_original"],
            "basis_sensitivity": anchored_nmse[worst_basis] / anchored_nmse[best_basis],
        },
    })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rec = run(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
