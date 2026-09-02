#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gaussian_shift_interface.run_c61r_seed import (
    adapt_anchored,
    build_base_hierarchy,
    canonical_nested_qr,
    fixed_calibration_indices,
    sha256_int_array,
    sha256_tensor,
    shifted_input,
)
from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import (
    FullSpanReplacedNet,
    accuracy,
    acts,
    compile_final_from_hierarchy,
    set_all_trainable,
    split_data,
    train_teacher,
)

FRESH_SEEDS = tuple(range(62400, 62416))
IMPLEMENTATION_VERIFICATION_SEED = 62300
GAUSSIAN_SIGMA = 0.04
CALIBRATION_SAMPLES = 192
FIT_UPDATES = 600
TOP_ADAPTATION_OFFSET = 645000
FINAL_INIT_OFFSET = 646000
FINAL_FIT_OFFSET = 647000
DIMENSIONS = (0, 1, 2)


def correction_for_k(residual: torch.Tensor, q: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return torch.zeros_like(residual)
    p = q[:, :k]
    return (residual @ p) @ p.T


def euclidean_capture(residual: torch.Tensor, correction: torch.Tensor) -> float:
    total = torch.sum(residual * residual)
    if float(total) <= 0.0:
        return float("nan")
    return float(torch.sum(correction * correction) / total)


def logit_l2_retained_ratio(
    residual: torch.Tensor,
    correction: torch.Tensor,
    head_weight: torch.Tensor,
) -> float:
    full = residual @ head_weight.T
    rem = (residual - correction) @ head_weight.T
    denom = torch.sum(full * full)
    if float(denom) <= 0.0:
        return float("nan")
    return float(torch.sum(rem * rem) / denom)


def fisher_quadratic_mean(delta_z: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    first = torch.sum(probabilities * delta_z * delta_z, dim=1)
    mean = torch.sum(probabilities * delta_z, dim=1)
    e = torch.clamp(first - mean * mean, min=0.0)
    return e.mean()


def fisher_retained_ratio(
    residual: torch.Tensor,
    correction: torch.Tensor,
    head_weight: torch.Tensor,
    teacher_probabilities: torch.Tensor,
) -> float:
    full = residual @ head_weight.T
    rem = (residual - correction) @ head_weight.T
    denom = fisher_quadratic_mean(full, teacher_probabilities)
    if float(denom) <= 0.0:
        return float("nan")
    return float(fisher_quadratic_mean(rem, teacher_probabilities) / denom)


def geometry_for_split(
    residual: torch.Tensor,
    q: torch.Tensor,
    head_weight: torch.Tensor,
    teacher_probabilities: torch.Tensor,
) -> dict:
    out = {}
    for k in DIMENSIONS:
        correction = correction_for_k(residual, q, k)
        out[f"p{k}"] = {
            "dimension": k,
            "euclidean_capture_fraction": euclidean_capture(residual, correction),
            "logit_l2_retained_ratio": logit_l2_retained_ratio(residual, correction, head_weight),
            "fisher_retained_ratio": fisher_retained_ratio(
                residual, correction, head_weight, teacher_probabilities
            ),
        }
    return out


def evaluate_positive_condition(
    teacher,
    base_hierarchy,
    q: torch.Tensor,
    k: int,
    a0c: torch.Tensor,
    a4c: torch.Tensor,
    a0v: torch.Tensor,
    a4v: torch.Tensor,
    Xv_shift: torch.Tensor,
    yv: torch.Tensor,
    denom: float,
    seed: int,
) -> dict:
    if k <= 0:
        raise ValueError("positive correction condition requires k > 0")
    hierarchy = copy.deepcopy(base_hierarchy)
    with torch.no_grad():
        baseline_c = base_hierarchy(a0c).detach()
        residual_c = a4c - baseline_c
        correction = correction_for_k(residual_c, q, k)
        hybrid_target = (baseline_c + correction).detach()

    adapt_anchored(hierarchy, a0c, hybrid_target, FIT_UPDATES, seed + TOP_ADAPTATION_OFFSET)
    set_all_trainable(hierarchy, False)
    final, metrics = compile_final_from_hierarchy(
        hierarchy,
        a0c,
        a0v,
        a4v,
        denom,
        seed + FINAL_INIT_OFFSET,
        seed + FINAL_FIT_OFFSET,
    )
    metrics["final_replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(final)), Xv_shift, yv
    )
    metrics["dimension"] = int(k)
    metrics["teacher_correction_applied"] = True
    metrics["top_boundary_adaptation_skipped"] = False
    return metrics


def run(seed: int, allow_verification_seed: bool = False) -> dict:
    torch.set_num_threads(1)
    if seed not in FRESH_SEEDS and not (
        allow_verification_seed and seed == IMPLEMENTATION_VERIFICATION_SEED
    ):
        raise ValueError(
            f"seed {seed} is not in locked fresh cohort {FRESH_SEEDS[0]}-{FRESH_SEEDS[-1]} "
            f"and is not verification seed {IMPLEMENTATION_VERIFICATION_SEED}"
        )

    Xt, yt, Xv, yv = split_data()
    teacher = train_teacher(seed, Xt, yt, 60)
    a0t, a1t, a2t, a3t, a4t = acts(teacher, Xt)
    base_hierarchy, budget = build_base_hierarchy(seed, a0t, a1t, a2t, a3t, a4t)

    cal_idx = fixed_calibration_indices(len(Xt))
    cal_t = torch.tensor(cal_idx, dtype=torch.long)
    Xc_shift = shifted_input(Xt[cal_t], seed + 640100)
    Xv_shift = shifted_input(Xv, seed + 640200)

    ac = acts(teacher, Xc_shift)
    av = acts(teacher, Xv_shift)
    a0c, a4c = ac[0], ac[-1]
    a0v, a4v = av[0], av[-1]
    denom = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    with torch.no_grad():
        baseline_c = base_hierarchy(a0c).detach()
        baseline_v = base_hierarchy(a0v).detach()
        residual_c = a4c - baseline_c
        residual_v = a4v - baseline_v
        teacher_prob_c = F.softmax(teacher(Xc_shift), dim=1)
        teacher_prob_v = F.softmax(teacher(Xv_shift), dim=1)
        head_weight = teacher.head.weight.detach()

    q = canonical_nested_qr(residual_c)

    frozen_final, frozen_metrics = compile_final_from_hierarchy(
        copy.deepcopy(base_hierarchy),
        a0c,
        a0v,
        a4v,
        denom,
        seed + FINAL_INIT_OFFSET,
        seed + FINAL_FIT_OFFSET,
    )
    frozen_metrics["final_replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(frozen_final)), Xv_shift, yv
    )

    p0 = dict(frozen_metrics)
    p0["dimension"] = 0
    p0["teacher_correction_applied"] = False
    p0["top_boundary_adaptation_skipped"] = True
    p1 = evaluate_positive_condition(
        teacher, base_hierarchy, q, 1, a0c, a4c, a0v, a4v,
        Xv_shift, yv, denom, seed
    )
    p2 = evaluate_positive_condition(
        teacher, base_hierarchy, q, 2, a0c, a4c, a0v, a4v,
        Xv_shift, yv, denom, seed
    )

    diagnostics = {
        "task_weighted_geometry": {
            "shifted_calibration": geometry_for_split(
                residual_c, q, head_weight, teacher_prob_c
            ),
            "shifted_validation": geometry_for_split(
                residual_v, q, head_weight, teacher_prob_v
            ),
        }
    }

    required = []
    for rec in (p0, p1, p2):
        required.extend([
            rec["final_replacement_val_acc"],
            rec["final_nmse_vs_original"],
        ])
    for split in diagnostics["task_weighted_geometry"].values():
        for rec in split.values():
            required.extend([
                rec["euclidean_capture_fraction"],
                rec["logit_l2_retained_ratio"],
                rec["fisher_retained_ratio"],
            ])
    finite = all(math.isfinite(float(x)) for x in required)
    positive_nmse = all(float(rec["final_nmse_vs_original"]) > 0.0 for rec in (p0, p1, p2))
    nonnegative_diagnostics = all(float(x) >= 0.0 for x in required[-18:])
    eligible = bool(
        budget["exact_4096_each_level"] and finite and positive_nmse and nonnegative_diagnostics
    )

    conditions = {"p0": p0, "p1": p1, "p2": p2}
    decision_row = {
        "seed": int(seed),
        "eligible": eligible,
        "conditions": {
            key: {
                "validation_accuracy": float(rec["final_replacement_val_acc"]),
                "nmse": float(rec["final_nmse_vs_original"]),
            }
            for key, rec in conditions.items()
        },
        "diagnostics": diagnostics,
    }

    return {
        "experiment": "C64R_P0_P1_P2_TASK_WEIGHTED_FRONTIER_EXPLORATION",
        "evidence_class": "PROSPECTIVE_EXPLORATORY",
        "status": "FRESH_SEED_OUTCOME" if seed in FRESH_SEEDS else "IMPLEMENTATION_VERIFICATION_OUTCOME",
        "seed": int(seed),
        "eligible": eligible,
        "test_evaluated": False,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "protocol": {
            "sigma": GAUSSIAN_SIGMA,
            "calibration_samples": CALIBRATION_SAMPLES,
            "dimensions": list(DIMENSIONS),
            "p0_definition": "frozen base hierarchy compiled directly; no teacher residual correction and no top-boundary adaptation",
            "top_adaptation_seed": seed + TOP_ADAPTATION_OFFSET,
            "final_init_seed": seed + FINAL_INIT_OFFSET,
            "final_fit_seed": seed + FINAL_FIT_OFFSET,
        },
        "provenance_hashes": {
            "calibration_indices_sha256": sha256_int_array(cal_idx),
            "shifted_calibration_tensor_sha256": sha256_tensor(Xc_shift),
            "shifted_validation_tensor_sha256": sha256_tensor(Xv_shift),
            "nested_q1_sha256": sha256_tensor(q[:, :1]),
            "nested_q2_sha256": sha256_tensor(q[:, :2]),
            "teacher_head_weight_sha256": sha256_tensor(head_weight),
        },
        "teacher_clean_validation_accuracy": accuracy(teacher, Xv, yv),
        "teacher_shifted_validation_accuracy": accuracy(teacher, Xv_shift, yv),
        "budget": budget,
        "conditions": conditions,
        "diagnostics": diagnostics,
        "decision_row": decision_row,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-verification-seed", action="store_true")
    args = ap.parse_args()
    rec = run(args.seed, allow_verification_seed=args.allow_verification_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
