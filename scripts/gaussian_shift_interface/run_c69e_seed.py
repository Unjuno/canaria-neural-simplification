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
)
from scripts.gaussian_shift_interface.run_c64r_seed import correction_for_k
from scripts.gaussian_shift_interface.run_c68e_seed import (
    state_dict_sha256,
    train_paired_teachers,
)
from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import (
    FullSpanReplacedNet,
    accuracy,
    acts,
    compile_final_from_hierarchy,
    set_all_trainable,
    split_data,
)

FRESH_SEEDS = tuple(range(67400, 67416))
IMPLEMENTATION_VERIFICATION_SEED = 67300
GAUSSIAN_SIGMA = 0.36
CALIBRATION_SAMPLES = 192
FIT_UPDATES = 600
CANDIDATE_DIMENSIONS = (0, 1, 2, 4, 8, 16)
REFERENCE_DIMENSION = 32
ALL_DIMENSIONS = CANDIDATE_DIMENSIONS + (REFERENCE_DIMENSION,)
CAL_EPS_OFFSET = 690100
VAL_EPS_OFFSET = 690200
TOP_ADAPTATION_OFFSET = 695000
FINAL_INIT_OFFSET = 696000
FINAL_FIT_OFFSET = 697000


def standard_normal_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(x.shape, generator=gen, dtype=x.dtype)


def shifted_input_from_epsilon(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x + GAUSSIAN_SIGMA * eps, 0.0, 1.0)


def evaluate_positive_condition(
    teacher,
    base_hierarchy,
    q: torch.Tensor,
    k: int,
    a0c: torch.Tensor,
    a4c: torch.Tensor,
    a0v: torch.Tensor,
    a4v: torch.Tensor,
    xv_shift: torch.Tensor,
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
    adapt_anchored(
        hierarchy,
        a0c,
        hybrid_target,
        FIT_UPDATES,
        seed + TOP_ADAPTATION_OFFSET,
    )
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
        FullSpanReplacedNet(teacher, copy.deepcopy(final)), xv_shift, yv
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

    xt, yt, xv, yv = split_data()
    clean_teacher, teacher, teacher_training_prov = train_paired_teachers(seed, xt, yt)
    robust_teacher_clean_acc = accuracy(teacher, xv, yv)
    clean_teacher_clean_acc = accuracy(clean_teacher, xv, yv)

    a0t, a1t, a2t, a3t, a4t = acts(teacher, xt)
    base_hierarchy, budget = build_base_hierarchy(seed, a0t, a1t, a2t, a3t, a4t)

    cal_idx = fixed_calibration_indices(len(xt))
    cal_t = torch.tensor(cal_idx, dtype=torch.long)
    xc_clean = xt[cal_t]
    cal_eps = standard_normal_like(xc_clean, seed + CAL_EPS_OFFSET)
    val_eps = standard_normal_like(xv, seed + VAL_EPS_OFFSET)
    xc_shift = shifted_input_from_epsilon(xc_clean, cal_eps)
    xv_shift = shifted_input_from_epsilon(xv, val_eps)

    ac = acts(teacher, xc_shift)
    av = acts(teacher, xv_shift)
    a0c, a4c = ac[0], ac[-1]
    a0v, a4v = av[0], av[-1]
    denom = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    with torch.no_grad():
        baseline_c = base_hierarchy(a0c).detach()
        residual_c = a4c - baseline_c
    q = canonical_nested_qr(residual_c)
    if q.shape[1] < REFERENCE_DIMENSION:
        raise AssertionError(("insufficient QR dimension", tuple(q.shape)))

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
        FullSpanReplacedNet(teacher, copy.deepcopy(frozen_final)), xv_shift, yv
    )
    p0 = dict(frozen_metrics)
    p0["dimension"] = 0
    p0["teacher_correction_applied"] = False
    p0["top_boundary_adaptation_skipped"] = True

    conditions: dict[str, dict] = {"p0": p0}
    for k in ALL_DIMENSIONS:
        if k == 0:
            continue
        conditions[f"p{k}"] = evaluate_positive_condition(
            teacher,
            base_hierarchy,
            q,
            k,
            a0c,
            a4c,
            a0v,
            a4v,
            xv_shift,
            yv,
            denom,
            seed,
        )

    robust_teacher_shifted_acc = accuracy(teacher, xv_shift, yv)
    clean_teacher_shifted_acc = accuracy(clean_teacher, xv_shift, yv)

    required = [
        robust_teacher_clean_acc,
        robust_teacher_shifted_acc,
        clean_teacher_clean_acc,
        clean_teacher_shifted_acc,
    ]
    for k in ALL_DIMENSIONS:
        rec = conditions[f"p{k}"]
        required.extend([
            rec["final_replacement_val_acc"],
            rec["final_nmse_vs_original"],
        ])
    finite = all(math.isfinite(float(x)) for x in required)
    positive_nmse = all(
        float(conditions[f"p{k}"]["final_nmse_vs_original"]) > 0.0
        for k in ALL_DIMENSIONS
    )
    eligible = bool(budget["exact_4096_each_level"] and finite and positive_nmse)

    decision_row = {
        "seed": int(seed),
        "eligible": eligible,
        "robust_teacher_clean_validation_accuracy": float(robust_teacher_clean_acc),
        "robust_teacher_shifted_validation_accuracy": float(robust_teacher_shifted_acc),
        "clean_teacher_clean_validation_accuracy": float(clean_teacher_clean_acc),
        "clean_teacher_shifted_validation_accuracy": float(clean_teacher_shifted_acc),
        "conditions": {
            f"p{k}": {
                "validation_accuracy": float(conditions[f"p{k}"]["final_replacement_val_acc"]),
                "nmse": float(conditions[f"p{k}"]["final_nmse_vs_original"]),
            }
            for k in ALL_DIMENSIONS
        },
    }

    return {
        "experiment": "C69E_ROBUST_TEACHER_INTERFACE_FRONTIER_EXPLORATION",
        "evidence_class": "PROSPECTIVE_EXPLORATORY",
        "status": "FRESH_SEED_OUTCOME" if seed in FRESH_SEEDS else "IMPLEMENTATION_VERIFICATION_OUTCOME",
        "seed": int(seed),
        "eligible": eligible,
        "test_evaluated": False,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "protocol": {
            "sigma": GAUSSIAN_SIGMA,
            "calibration_samples": CALIBRATION_SAMPLES,
            "candidate_dimensions": list(CANDIDATE_DIMENSIONS),
            "reference_dimension": REFERENCE_DIMENSION,
            "calibration_epsilon_seed": seed + CAL_EPS_OFFSET,
            "validation_epsilon_seed": seed + VAL_EPS_OFFSET,
            "top_adaptation_seed": seed + TOP_ADAPTATION_OFFSET,
            "final_init_seed": seed + FINAL_INIT_OFFSET,
            "final_fit_seed": seed + FINAL_FIT_OFFSET,
            "p0_definition": "frozen repaired-teacher base hierarchy compiled directly; no teacher residual correction and no top-boundary adaptation",
            "p32_role": "strong reference only; not claimed exact/full reconstruction",
        },
        "teacher_training_provenance": teacher_training_prov,
        "provenance_hashes": {
            "calibration_indices_sha256": sha256_int_array(cal_idx),
            "calibration_standard_normal_sha256": sha256_tensor(cal_eps),
            "validation_standard_normal_sha256": sha256_tensor(val_eps),
            "shifted_calibration_tensor_sha256": sha256_tensor(xc_shift),
            "shifted_validation_tensor_sha256": sha256_tensor(xv_shift),
            "robust_teacher_final_state_sha256": state_dict_sha256(teacher),
            "paired_clean_teacher_final_state_sha256": state_dict_sha256(clean_teacher),
            **{
                f"nested_q{k}_sha256": sha256_tensor(q[:, :k])
                for k in (1, 2, 4, 8, 16, 32)
            },
        },
        "budget": budget,
        "teacher_metrics": {
            "robust_clean_validation_accuracy": float(robust_teacher_clean_acc),
            "robust_shifted_validation_accuracy": float(robust_teacher_shifted_acc),
            "clean_paired_clean_validation_accuracy": float(clean_teacher_clean_acc),
            "clean_paired_shifted_validation_accuracy": float(clean_teacher_shifted_acc),
        },
        "conditions": conditions,
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
