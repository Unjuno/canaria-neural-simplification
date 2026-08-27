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

from scripts.exploration.c6_two_level_hierarchy import (
    Cluster,
    TinyRes,
    WholeReplacedNet,
    accuracy,
    acts,
    count_params,
    fit_map,
    nmse,
    set_all_trainable,
    set_seed,
    train_teacher,
)


def split_data_with_test_index():
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.25, random_state=1234, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.2,
        random_state=5678,
        stratify=y[train_idx],
    )
    return (
        torch.tensor(X[train_idx]),
        torch.tensor(y[train_idx], dtype=torch.long),
        torch.tensor(X[val_idx]),
        torch.tensor(y[val_idx], dtype=torch.long),
        test_idx.tolist(),
    )


def materialize_test(test_idx):
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    ix = np.asarray(test_idx, dtype=np.int64)
    return torch.tensor(X[ix]), torch.tensor(y[ix], dtype=torch.long)


def final_fit(seed, target_t, target_v, a0t, a0v, a4v, original_denom, teacher, Xv, yv):
    single = TinyRes(64, 32, seed + 160001)
    single = fit_map(single, a0t, target_t, 600, seed + 170001)
    with torch.no_grad():
        out = single(a0v)
    target_denom = float(((target_v - target_v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    return single, {
        "nmse_vs_immediate_canaria_teacher": nmse(out, target_v, target_denom),
        "nmse_vs_original_teacher": nmse(out, a4v, original_denom),
        "val_acc": accuracy(WholeReplacedNet(teacher, copy.deepcopy(single)), Xv, yv),
    }


def run(seed: int):
    set_seed(seed)
    Xt, yt, Xv, yv, test_idx = split_data_with_test_index()
    teacher = train_teacher(seed, Xt, yt)
    teacher_val = accuracy(teacher, Xv, yv)
    result = {
        "seed": seed,
        "status": "CONFIRMATORY_SEED_OUTCOME",
        "teacher_val_acc": teacher_val,
        "eligible": bool(teacher_val >= 0.95),
        "test_evaluated": False,
    }
    if not result["eligible"]:
        return result

    at = acts(teacher, Xt)
    av = acts(teacher, Xv)
    a0t, a1t, a2t, a3t, a4t = at
    a0v, _a1v, a2v, _a3v, a4v = av
    original_denom = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    locals4 = [
        fit_map(TinyRes(64, 8, seed + 101001), a0t, a1t, 600, seed + 102001),
        fit_map(TinyRes(64, 8, seed + 101002), a1t, a2t, 600, seed + 102002),
        fit_map(TinyRes(64, 8, seed + 101003), a2t, a3t, 600, seed + 102003),
        fit_map(TinyRes(64, 8, seed + 101004), a3t, a4t, 600, seed + 102004),
    ]

    pair12 = Cluster([copy.deepcopy(locals4[0]), copy.deepcopy(locals4[1])])
    pair34 = Cluster([copy.deepcopy(locals4[2]), copy.deepcopy(locals4[3])])
    set_all_trainable(pair12, True)
    set_all_trainable(pair34, True)
    pair12 = fit_map(pair12, a0t, a2t, 600, seed + 120001)
    pair34 = fit_map(pair34, a2t, a4t, 600, seed + 120002)
    set_all_trainable(pair12, False)
    set_all_trainable(pair34, False)

    with torch.no_grad():
        pair12_t = pair12(a0t).detach()
        pair12_v = pair12(a0v).detach()
        pair34_t = pair34(a2t).detach()
        lower_ir_t = pair34(pair12_t).detach()
        lower_ir_v = pair34(pair12_v).detach()

    c12 = fit_map(TinyRes(64, 16, seed + 130001), a0t, pair12_t, 600, seed + 140001)
    c34 = fit_map(TinyRes(64, 16, seed + 130002), a2t, pair34_t, 600, seed + 140002)
    with torch.no_grad():
        pre_t = c34(c12(a0t)).detach()
        pre_v = c34(c12(a0v)).detach()

    level2_pair = Cluster([copy.deepcopy(c12), copy.deepcopy(c34)])
    set_all_trainable(level2_pair, True)
    level2_pair = fit_map(level2_pair, a0t, lower_ir_t, 600, seed + 150001)
    set_all_trainable(level2_pair, False)
    with torch.no_grad():
        post_t = level2_pair(a0t).detach()
        post_v = level2_pair(a0v).detach()

    fitted = {}
    metrics = {}
    fitted["hierarchical_no_level2_adapt"], metrics["hierarchical_no_level2_adapt"] = final_fit(
        seed, pre_t, pre_v, a0t, a0v, a4v, original_denom, teacher, Xv, yv
    )
    fitted["hierarchical_level2_joint_adapt"], metrics["hierarchical_level2_joint_adapt"] = final_fit(
        seed, post_t, post_v, a0t, a0v, a4v, original_denom, teacher, Xv, yv
    )
    fitted["flat_lower_ir_single"], metrics["flat_lower_ir_single"] = final_fit(
        seed, lower_ir_t, lower_ir_v, a0t, a0v, a4v, original_denom, teacher, Xv, yv
    )
    fitted["direct_original_single"], metrics["direct_original_single"] = final_fit(
        seed, a4t, a4v, a0t, a0v, a4v, original_denom, teacher, Xv, yv
    )

    lower_ir_denom = float(((lower_ir_v - lower_ir_v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    with torch.no_grad():
        post_v_now = level2_pair(a0v)
    structural = {
        "lower_ir_nmse_vs_original": nmse(lower_ir_v, a4v, original_denom),
        "post_level2_pair_nmse_vs_lower_ir": nmse(post_v_now, lower_ir_v, lower_ir_denom),
        "post_level2_pair_nmse_vs_original": nmse(post_v_now, a4v, original_denom),
    }

    # Test is materialized only after every locked fit and validation metric above is complete.
    Xte, yte = materialize_test(test_idx)
    result["test_evaluated"] = True
    result["teacher_test_acc"] = accuracy(teacher, Xte, yte)
    for name, single in fitted.items():
        metrics[name]["test_acc"] = accuracy(
            WholeReplacedNet(teacher, copy.deepcopy(single)), Xte, yte
        )

    joint = metrics["hierarchical_level2_joint_adapt"]
    no_adapt = metrics["hierarchical_no_level2_adapt"]
    flat = metrics["flat_lower_ir_single"]
    direct = metrics["direct_original_single"]
    result.update({
        "budget": {
            "level0_total_params": sum(count_params(m) for m in locals4),
            "level1_total_params": count_params(c12) + count_params(c34),
            "level2_params": count_params(fitted["hierarchical_level2_joint_adapt"]),
            "exact_4096_match": sum(count_params(m) for m in locals4) == count_params(c12) + count_params(c34) == count_params(fitted["hierarchical_level2_joint_adapt"]) == 4096,
        },
        "structural": structural,
        "conditions": metrics,
        "derived": {
            "D_joint_minus_no_adapt": joint["nmse_vs_original_teacher"] - no_adapt["nmse_vs_original_teacher"],
            "R_joint_over_flat": joint["nmse_vs_original_teacher"] / flat["nmse_vs_original_teacher"],
            "R_joint_over_direct": joint["nmse_vs_original_teacher"] / direct["nmse_vs_original_teacher"],
            "val_acc_diff_joint_minus_flat": joint["val_acc"] - flat["val_acc"],
            "test_acc_diff_joint_minus_flat": joint["test_acc"] - flat["test_acc"],
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
