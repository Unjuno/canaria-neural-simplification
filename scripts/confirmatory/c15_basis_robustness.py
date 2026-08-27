from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exploration.c10_boundary_signal_ablation import (
    Chain, FullSpanReplacedNet, TinyRes, accuracy, acts,
    compile_final_from_hierarchy, count_params, fit_map, nmse,
    set_all_trainable, train_teacher,
)
from scripts.exploration.c12_self_anchored_sketches import adapt_anchored
from scripts.exploration.c14_basis_robustness import random_orthogonal_basis


def split_data_and_test_index():
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=1234, stratify=y)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=5678, stratify=y[train_idx])
    return (
        torch.tensor(X[train_idx]), torch.tensor(y[train_idx], dtype=torch.long),
        torch.tensor(X[val_idx]), torch.tensor(y[val_idx], dtype=torch.long),
        test_idx.tolist(),
    )


def materialize_test(test_idx):
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    ix = np.asarray(test_idx, dtype=np.int64)
    return torch.tensor(X[ix]), torch.tensor(y[ix], dtype=torch.long)


def basis_family():
    eye = torch.eye(64, dtype=torch.float32)
    return {
        "identity_first32": eye[:, :32],
        "random_20260920": random_orthogonal_basis(20260920)[:, :32],
        "random_20260921": random_orthogonal_basis(20260921)[:, :32],
        "random_20260922": random_orthogonal_basis(20260922)[:, :32],
    }


def run(seed: int):
    Xt, yt, Xv, yv, test_idx = split_data_and_test_index()
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
    fitted = {}

    frozen = copy.deepcopy(base_hierarchy)
    frozen_final, frozen_metrics = compile_final_from_hierarchy(
        frozen, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
    )
    frozen_metrics["final_replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(frozen_final)), Xv, yv
    )
    conditions["frozen"] = frozen_metrics

    basis_names = list(basis_family().keys())
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
        fitted[name] = copy.deepcopy(final)

    full = copy.deepcopy(base_hierarchy)
    adapt_anchored(full, a0t, a4t.detach(), 600, top_seed)
    set_all_trainable(full, False)
    full_final, full_metrics = compile_final_from_hierarchy(
        full, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
    )
    full_metrics["final_replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(full_final)), Xv, yv
    )
    conditions["full_64"] = full_metrics
    fitted["full_64"] = copy.deepcopy(full_final)

    direct = TinyRes(64, 32, final_init_seed)
    direct = fit_map(direct, a0t, a4t, 600, final_fit_seed)
    with torch.no_grad():
        direct_v = direct(a0v)
    direct_nmse = nmse(direct_v, a4v, denom_full)
    direct_val_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(direct)), Xv, yv)

    basis_nmse = np.array([conditions[n]["final_nmse_vs_original"] for n in basis_names])
    basis_val_acc = np.array([conditions[n]["final_replacement_val_acc"] for n in basis_names])
    frozen_nmse = conditions["frozen"]["final_nmse_vs_original"]
    full_nmse = conditions["full_64"]["final_nmse_vs_original"]
    derived = {
        "D_worst": float(basis_nmse.max() - frozen_nmse),
        "R_worst": float(basis_nmse.max() / full_nmse),
        "basis_spread": float(basis_nmse.max() / basis_nmse.min()),
        "val_acc_diff_worst_minus_full": float(basis_val_acc.min() - conditions["full_64"]["final_replacement_val_acc"]),
    }

    Xte, yte = materialize_test(test_idx)
    teacher_test_acc = accuracy(teacher, Xte, yte)
    for name, model in fitted.items():
        conditions[name]["final_replacement_test_acc"] = accuracy(
            FullSpanReplacedNet(teacher, copy.deepcopy(model)), Xte, yte
        )
    basis_test_acc = np.array([conditions[n]["final_replacement_test_acc"] for n in basis_names])
    derived["test_acc_diff_worst_minus_full"] = float(
        basis_test_acc.min() - conditions["full_64"]["final_replacement_test_acc"]
    )

    return {
        "seed": seed,
        "status": "CONFIRMATORY_SEED_OUTCOME",
        "test_evaluated_after_validation_only": True,
        "teacher_val_acc": accuracy(teacher, Xv, yv),
        "teacher_test_acc": teacher_test_acc,
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
            "replacement_val_acc": direct_val_acc,
        },
        "derived": derived,
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
