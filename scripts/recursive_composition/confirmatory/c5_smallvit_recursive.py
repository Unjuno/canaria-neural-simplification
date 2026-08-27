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

from scripts.exploration.c4_smallvit_recursive import (
    Cluster,
    ReplacedViT,
    SmallViT,
    TinyTokenRes,
    accuracy,
    collect_span,
    count_params,
    fit_map,
    nmse,
    set_seed,
    set_trainable,
    train_teacher,
)


def split_data_and_test_index():
    d = load_digits()
    X = (d.images.astype(np.float32) / 16.0)[:, None, :, :]
    y = d.target.astype(np.int64)
    idx = np.arange(len(y))
    tr, temp = train_test_split(idx, test_size=0.30, random_state=24680, stratify=y)
    va, te = train_test_split(temp, test_size=0.50, random_state=13579, stratify=y[temp])

    def ds(ii):
        return torch.utils.data.TensorDataset(torch.from_numpy(X[ii]), torch.from_numpy(y[ii]))

    return ds(tr), ds(va), te.tolist()


def materialize_test(test_idx):
    d = load_digits()
    X = (d.images.astype(np.float32) / 16.0)[:, None, :, :]
    y = d.target.astype(np.int64)
    ix = np.asarray(test_idx, dtype=np.int64)
    return torch.utils.data.TensorDataset(torch.from_numpy(X[ix]), torch.from_numpy(y[ix]))


def run(seed: int):
    set_seed(seed)
    tr, va, test_idx = split_data_and_test_index()
    teacher = SmallViT()
    teacher = train_teacher(teacher, tr, seed + 50000, 45)
    teacher_val = accuracy(teacher, va)
    result = {
        "seed": seed,
        "status": "CONFIRMATORY_SEED_OUTCOME",
        "teacher_val_acc": teacher_val,
        "eligible": bool(teacher_val >= 0.95),
        "test_evaluated": False,
    }
    if not result["eligible"]:
        return result

    a0t, a1t, a2t = collect_span(teacher, tr, start=1)
    a0v, _a1v, a2v = collect_span(teacher, va, start=1)
    a0fit, a1fit, a2fit = a0t[:512], a1t[:512], a2t[:512]
    teacher_denom = float(((a2v - a2v.mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12

    c1 = fit_map(TinyTokenRes(32, 32, seed + 101001), a0fit, a1fit, 600, seed + 102001)
    c2 = fit_map(TinyTokenRes(32, 32, seed + 101002), a1fit, a2fit, 600, seed + 102002)
    base_cluster = Cluster([c1, c2])

    stage2_seed = seed + 120000
    stage3_init_seed = seed + 130000
    stage3_fit_seed = seed + 140000
    conditions = {}
    fitted = {}

    schedules = {
        "all_frozen_recursive": [],
        "all_unfrozen_recursive": [0, 1],
    }

    for name, trainable in schedules.items():
        cluster = copy.deepcopy(base_cluster)
        set_trainable(cluster, trainable)
        if trainable:
            fit_map(cluster, a0fit, a2fit, 600, stage2_seed)
        set_trainable(cluster, [])
        with torch.no_grad():
            cluster_fit_target = cluster(a0fit).detach()
            cluster_val_target = cluster(a0v).detach()
        cluster_val_nmse = nmse(cluster_val_target, a2v, teacher_denom)
        cluster_val_acc = accuracy(ReplacedViT(teacher, copy.deepcopy(cluster), 1), va)

        single = TinyTokenRes(32, 64, stage3_init_seed)
        single = fit_map(single, a0fit, cluster_fit_target, 600, stage3_fit_seed)
        with torch.no_grad():
            single_val = single(a0v)
        cluster_denom = float(
            ((cluster_val_target - cluster_val_target.mean(dim=(0, 1), keepdim=True)) ** 2).mean()
        ) + 1e-12
        single_val_acc = accuracy(ReplacedViT(teacher, copy.deepcopy(single), 1), va)
        conditions[name] = {
            "cluster_val_nmse_vs_original": cluster_val_nmse,
            "cluster_val_acc": cluster_val_acc,
            "recursive_val_nmse_vs_cluster": nmse(single_val, cluster_val_target, cluster_denom),
            "recursive_val_nmse_vs_original": nmse(single_val, a2v, teacher_denom),
            "recursive_val_acc": single_val_acc,
        }
        fitted[name] = copy.deepcopy(single)

    direct = TinyTokenRes(32, 64, stage3_init_seed)
    direct = fit_map(direct, a0fit, a2fit, 600, stage3_fit_seed)
    with torch.no_grad():
        direct_val = direct(a0v)
    direct_val_nmse = nmse(direct_val, a2v, teacher_denom)
    direct_val_acc = accuracy(ReplacedViT(teacher, copy.deepcopy(direct), 1), va)

    # Held-out test is materialized only after every locked fit and validation metric above is complete.
    te = materialize_test(test_idx)
    teacher_test = accuracy(teacher, te)
    for name in schedules:
        conditions[name]["recursive_test_acc"] = accuracy(
            ReplacedViT(teacher, copy.deepcopy(fitted[name]), 1), te
        )
    direct_test_acc = accuracy(ReplacedViT(teacher, copy.deepcopy(direct), 1), te)

    result.update(
        {
            "test_evaluated": True,
            "teacher_test_acc": teacher_test,
            "cluster_params": count_params(base_cluster),
            "single_params": count_params(direct),
            "exact_parameter_match": count_params(base_cluster) == count_params(direct) == 4096,
            "conditions": conditions,
            "direct_original_single": {
                "val_nmse_vs_original": direct_val_nmse,
                "val_acc": direct_val_acc,
                "test_acc": direct_test_acc,
            },
            "derived": {
                "D_joint_minus_frozen": conditions["all_unfrozen_recursive"]["recursive_val_nmse_vs_original"]
                - conditions["all_frozen_recursive"]["recursive_val_nmse_vs_original"],
                "R_joint_over_direct": conditions["all_unfrozen_recursive"]["recursive_val_nmse_vs_original"]
                / direct_val_nmse,
                "val_acc_diff_joint_minus_direct": conditions["all_unfrozen_recursive"]["recursive_val_acc"]
                - direct_val_acc,
                "test_acc_diff_joint_minus_direct": conditions["all_unfrozen_recursive"]["recursive_test_acc"]
                - direct_test_acc,
            },
        }
    )
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
