from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exploration.c10_boundary_signal_ablation import (
    Chain,
    FullSpanReplacedNet,
    TinyRes,
    accuracy,
    acts,
    compile_final_from_hierarchy,
    count_params,
    fit_map,
    nmse,
    set_all_trainable,
    split_data,
    train_teacher,
)
from scripts.exploration.c12_self_anchored_sketches import adapt_anchored


def random_orthogonal_basis(seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((64, 64))
    q, r = np.linalg.qr(a)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    q = q * signs[None, :]
    return torch.tensor(q.astype(np.float32))


def basis_family():
    eye = torch.eye(64, dtype=torch.float32)
    return {
        "identity_first32": eye[:, :32],
        "random_20260910": random_orthogonal_basis(20260910)[:, :32],
        "random_20260911": random_orthogonal_basis(20260911)[:, :32],
        "random_20260912": random_orthogonal_basis(20260912)[:, :32],
    }


def run(seed: int):
    Xt, yt, Xv, yv = split_data()
    teacher = train_teacher(seed, Xt, yt, 60)
    at = acts(teacher, Xt)
    av = acts(teacher, Xv)
    a0t, a1t, a2t, a3t, a4t = at
    a0v, _a1v, a2v, _a3v, a4v = av
    denom_full = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    locals4 = [
        fit_map(TinyRes(64, 8, seed + 101001), a0t, a1t, 600, seed + 102001),
        fit_map(TinyRes(64, 8, seed + 101002), a1t, a2t, 600, seed + 102002),
        fit_map(TinyRes(64, 8, seed + 101003), a2t, a3t, 600, seed + 102003),
        fit_map(TinyRes(64, 8, seed + 101004), a3t, a4t, 600, seed + 102004),
    ]

    pair12 = Chain([copy.deepcopy(locals4[0]), copy.deepcopy(locals4[1])])
    set_all_trainable(pair12, True)
    fit_map(pair12, a0t, a2t, 600, seed + 110001)
    set_all_trainable(pair12, False)

    pair34 = Chain([copy.deepcopy(locals4[2]), copy.deepcopy(locals4[3])])
    set_all_trainable(pair34, True)
    fit_map(pair34, a2t, a4t, 600, seed + 110002)
    set_all_trainable(pair34, False)

    with torch.no_grad():
        pair12_t = pair12(a0t).detach()
        pair34_t = pair34(a2t).detach()
    c12 = fit_map(TinyRes(64, 16, seed + 120001), a0t, pair12_t, 600, seed + 121001)
    c34 = fit_map(TinyRes(64, 16, seed + 120002), a2t, pair34_t, 600, seed + 121002)

    base_hierarchy = Chain([copy.deepcopy(c12), copy.deepcopy(c34)])
    set_all_trainable(base_hierarchy, False)
    with torch.no_grad():
        baseline_t = base_hierarchy(a0t).detach()

    top_seed = seed + 150000
    final_init_seed = seed + 160000
    final_fit_seed = seed + 170000

    conditions = {}

    frozen = copy.deepcopy(base_hierarchy)
    final, metrics = compile_final_from_hierarchy(
        frozen, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
    )
    metrics["final_replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(final)), Xv, yv
    )
    conditions["frozen"] = metrics

    for name, p in basis_family().items():
        hierarchy = copy.deepcopy(base_hierarchy)
        with torch.no_grad():
            correction = ((a4t - baseline_t) @ p) @ p.T
            hybrid_target = (baseline_t + correction).detach()
        adapt_anchored(hierarchy, a0t, hybrid_target, 600, top_seed)
        set_all_trainable(hierarchy, False)
        final, metrics = compile_final_from_hierarchy(
            hierarchy, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
        )
        metrics["final_replacement_val_acc"] = accuracy(
            FullSpanReplacedNet(teacher, copy.deepcopy(final)), Xv, yv
        )
        conditions[name] = metrics

    full = copy.deepcopy(base_hierarchy)
    adapt_anchored(full, a0t, a4t.detach(), 600, top_seed)
    set_all_trainable(full, False)
    final_full, full_metrics = compile_final_from_hierarchy(
        full, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
    )
    full_metrics["final_replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(final_full)), Xv, yv
    )
    conditions["full_64"] = full_metrics

    direct = TinyRes(64, 32, final_init_seed)
    direct = fit_map(direct, a0t, a4t, 600, final_fit_seed)
    with torch.no_grad():
        direct_v = direct(a0v)
    direct_nmse = nmse(direct_v, a4v, denom_full)
    direct_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(direct)), Xv, yv)

    frozen_nmse = conditions["frozen"]["final_nmse_vs_original"]
    full_nmse = conditions["full_64"]["final_nmse_vs_original"]
    basis_names = list(basis_family().keys())
    for name in basis_names:
        rec = conditions[name]
        rec["difference_vs_frozen"] = rec["final_nmse_vs_original"] - frozen_nmse
        rec["ratio_over_full64"] = rec["final_nmse_vs_original"] / full_nmse

    basis_nmse = [conditions[n]["final_nmse_vs_original"] for n in basis_names]
    return {
        "seed": seed,
        "status": "EXPLORATORY_OUTCOME",
        "test_evaluated": False,
        "teacher_val_acc": accuracy(teacher, Xv, yv),
        "budget": {
            "local_total_params": sum(count_params(m) for m in locals4),
            "level1_total_params": count_params(c12) + count_params(c34),
            "final_params": count_params(direct),
            "exact_4096_each_level": (
                sum(count_params(m) for m in locals4)
                == count_params(c12) + count_params(c34)
                == count_params(direct)
                == 4096
            ),
        },
        "basis_names": basis_names,
        "conditions": conditions,
        "direct_original_single": {
            "final_nmse_vs_original": direct_nmse,
            "replacement_val_acc": direct_acc,
        },
        "basis_summary": {
            "best_nmse": min(basis_nmse),
            "worst_nmse": max(basis_nmse),
            "worst_over_best": max(basis_nmse) / min(basis_nmse),
            "all_improve_frozen": all(conditions[n]["difference_vs_frozen"] < 0 for n in basis_names),
            "all_ratio_over_full64": {n: conditions[n]["ratio_over_full64"] for n in basis_names},
        },
        "ordering_basis_nmse_best_to_worst": sorted(
            basis_names, key=lambda n: conditions[n]["final_nmse_vs_original"]
        ),
    }


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
