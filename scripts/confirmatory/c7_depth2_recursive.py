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

from scripts.exploration.c6_hierarchical_recursive import (
    Chain,
    FullSpanReplacedNet,
    TinyRes,
    Net,
    accuracy,
    acts,
    count_params,
    final_from_recursive_teacher,
    fit_map,
    nmse,
    set_all_trainable,
    train_teacher,
)


def split_data_and_test_index():
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.25, random_state=1234, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=5678, stratify=y[train_idx]
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


def run(seed):
    Xt, yt, Xv, yv, test_idx = split_data_and_test_index()
    teacher = train_teacher(seed, Xt, yt, 60)
    at = acts(teacher, Xt)
    av = acts(teacher, Xv)
    a0t, a1t, a2t, a3t, a4t = at
    a0v, a1v, a2v, a3v, a4v = av
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
    final_init_seed = seed + 130000
    final_fit_seed = seed + 140000
    conditions = {}
    fitted = {}

    hierarchy_frozen = copy.deepcopy(base_hierarchy)
    final_hf, m_hf = final_from_recursive_teacher(
        seed, hierarchy_frozen, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
    )
    m_hf["replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(final_hf)), Xv, yv
    )
    conditions["hierarchical_frozen"] = m_hf
    fitted["hierarchical_frozen"] = copy.deepcopy(final_hf)

    hierarchy_joint = copy.deepcopy(base_hierarchy)
    set_all_trainable(hierarchy_joint, True)
    fit_map(hierarchy_joint, a0t, a4t, 600, seed + 150000)
    set_all_trainable(hierarchy_joint, False)
    final_hj, m_hj = final_from_recursive_teacher(
        seed, hierarchy_joint, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
    )
    m_hj["replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(final_hj)), Xv, yv
    )
    conditions["hierarchical_joint"] = m_hj
    fitted["hierarchical_joint"] = copy.deepcopy(final_hj)

    single_level_cluster = Chain([copy.deepcopy(m) for m in locals4])
    set_all_trainable(single_level_cluster, True)
    fit_map(single_level_cluster, a0t, a4t, 600, seed + 160000)
    set_all_trainable(single_level_cluster, False)
    final_sl, m_sl = final_from_recursive_teacher(
        seed, single_level_cluster, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
    )
    m_sl["replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(final_sl)), Xv, yv
    )
    conditions["single_level_recursive"] = m_sl
    fitted["single_level_recursive"] = copy.deepcopy(final_sl)

    direct = TinyRes(64, 32, final_init_seed)
    direct = fit_map(direct, a0t, a4t, 600, final_fit_seed)
    with torch.no_grad():
        direct_v = direct(a0v)
    direct_nmse = nmse(direct_v, a4v, denom_full)
    direct_val_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(direct)), Xv, yv)

    for rec in conditions.values():
        rec["ratio_final_nmse_over_direct"] = rec["final_nmse_vs_original"] / direct_nmse

    validation_complete = {
        "D_frozen": conditions["hierarchical_joint"]["final_nmse_vs_original"]
        - conditions["hierarchical_frozen"]["final_nmse_vs_original"],
        "R_joint": conditions["hierarchical_joint"]["final_nmse_vs_original"] / direct_nmse,
        "D_single": conditions["hierarchical_joint"]["final_nmse_vs_original"]
        - conditions["single_level_recursive"]["final_nmse_vs_original"],
        "R_strict": conditions["hierarchical_frozen"]["final_nmse_vs_original"] / direct_nmse,
        "val_acc_diff_joint_minus_direct": conditions["hierarchical_joint"]["replacement_val_acc"]
        - direct_val_acc,
    }

    # Held-out test is materialized only after all locked fitting and validation metrics above are complete.
    Xte, yte = materialize_test(test_idx)
    teacher_test_acc = accuracy(teacher, Xte, yte)
    for name in fitted:
        conditions[name]["replacement_test_acc"] = accuracy(
            FullSpanReplacedNet(teacher, copy.deepcopy(fitted[name])), Xte, yte
        )
    direct_test_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(direct)), Xte, yte)
    validation_complete["test_acc_diff_joint_minus_direct"] = (
        conditions["hierarchical_joint"]["replacement_test_acc"] - direct_test_acc
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
        "conditions": conditions,
        "direct_original_single": {
            "final_nmse_vs_original": direct_nmse,
            "replacement_val_acc": direct_val_acc,
            "replacement_test_acc": direct_test_acc,
        },
        "derived": validation_complete,
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
