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
from scripts.gaussian_shift_interface.run_c68e_seed import state_dict_sha256, train_paired_teachers
from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import (
    FullSpanReplacedNet,
    accuracy,
    acts,
    compile_final_from_hierarchy,
    set_all_trainable,
    split_data,
)

FRESH_SEEDS = tuple(range(68400, 68416))
IMPLEMENTATION_VERIFICATION_SEED = 68300
GAUSSIAN_SIGMA = 0.36
CALIBRATION_SAMPLES = 192
FIT_UPDATES = 600
DIMENSIONS = (32, 64)
CAL_EPS_OFFSET = 700100
VAL_EPS_OFFSET = 700200
TOP_ADAPTATION_OFFSET = 705000
FINAL_INIT_OFFSET = 706000
FINAL_FIT_OFFSET = 707000
FULL_BASIS_REL_SQERR_MAX = 1e-10


def standard_normal_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(x.shape, generator=gen, dtype=x.dtype)


def shifted_input(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x + GAUSSIAN_SIGMA * eps, 0.0, 1.0)


def evaluate_condition(
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
    robust_clean_acc = accuracy(teacher, xv, yv)

    a0t, a1t, a2t, a3t, a4t = acts(teacher, xt)
    base_hierarchy, budget = build_base_hierarchy(seed, a0t, a1t, a2t, a3t, a4t)

    cal_idx = fixed_calibration_indices(len(xt))
    cal_t = torch.tensor(cal_idx, dtype=torch.long)
    xc_clean = xt[cal_t]
    cal_eps = standard_normal_like(xc_clean, seed + CAL_EPS_OFFSET)
    val_eps = standard_normal_like(xv, seed + VAL_EPS_OFFSET)
    xc_shift = shifted_input(xc_clean, cal_eps)
    xv_shift = shifted_input(xv, val_eps)

    ac = acts(teacher, xc_shift)
    av = acts(teacher, xv_shift)
    a0c, a4c = ac[0], ac[-1]
    a0v, a4v = av[0], av[-1]
    denom = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    with torch.no_grad():
        baseline_c = base_hierarchy(a0c).detach()
        residual_c = a4c - baseline_c
    q = canonical_nested_qr(residual_c)
    if q.shape[1] < 64:
        raise AssertionError(("insufficient QR dimension", tuple(q.shape)))
    with torch.no_grad():
        full_reconstruction = correction_for_k(residual_c, q, 64)
        rel_sqerr = float(
            ((residual_c - full_reconstruction) ** 2).sum()
            / (((residual_c) ** 2).sum() + 1e-30)
        )

    conditions = {
        f"p{k}": evaluate_condition(
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
        for k in DIMENSIONS
    }

    robust_shifted_acc = accuracy(teacher, xv_shift, yv)
    clean_teacher_clean_acc = accuracy(clean_teacher, xv, yv)
    clean_teacher_shifted_acc = accuracy(clean_teacher, xv_shift, yv)

    required = [robust_clean_acc, robust_shifted_acc, clean_teacher_clean_acc, clean_teacher_shifted_acc, rel_sqerr]
    for k in DIMENSIONS:
        required.extend([
            conditions[f"p{k}"]["final_replacement_val_acc"],
            conditions[f"p{k}"]["final_nmse_vs_original"],
        ])
    finite = all(math.isfinite(float(x)) for x in required)
    positive_nmse = all(float(conditions[f"p{k}"]["final_nmse_vs_original"]) > 0.0 for k in DIMENSIONS)
    full_basis_ok = bool(rel_sqerr <= FULL_BASIS_REL_SQERR_MAX)
    eligible = bool(budget["exact_4096_each_level"] and finite and positive_nmse and full_basis_ok)

    decision_row = {
        "seed": int(seed),
        "eligible": eligible,
        "full_basis_relative_calibration_residual_sqerr": rel_sqerr,
        "robust_teacher_clean_validation_accuracy": float(robust_clean_acc),
        "robust_teacher_shifted_validation_accuracy": float(robust_shifted_acc),
        "clean_teacher_clean_validation_accuracy": float(clean_teacher_clean_acc),
        "clean_teacher_shifted_validation_accuracy": float(clean_teacher_shifted_acc),
        "p32_validation_accuracy": float(conditions["p32"]["final_replacement_val_acc"]),
        "p64_validation_accuracy": float(conditions["p64"]["final_replacement_val_acc"]),
        "p32_nmse": float(conditions["p32"]["final_nmse_vs_original"]),
        "p64_nmse": float(conditions["p64"]["final_nmse_vs_original"]),
    }

    return {
        "experiment": "C70E_P64_REFERENCE_REPAIR_EXPLORATION",
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
            "calibration_epsilon_seed": seed + CAL_EPS_OFFSET,
            "validation_epsilon_seed": seed + VAL_EPS_OFFSET,
            "top_adaptation_seed": seed + TOP_ADAPTATION_OFFSET,
            "final_init_seed": seed + FINAL_INIT_OFFSET,
            "final_fit_seed": seed + FINAL_FIT_OFFSET,
            "full_basis_relative_sqerr_threshold": FULL_BASIS_REL_SQERR_MAX,
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
            "nested_q32_sha256": sha256_tensor(q[:, :32]),
            "nested_q64_sha256": sha256_tensor(q[:, :64]),
        },
        "budget": budget,
        "implementation_invariant": {
            "full_basis_relative_calibration_residual_sqerr": rel_sqerr,
            "threshold": FULL_BASIS_REL_SQERR_MAX,
            "pass": full_basis_ok,
        },
        "teacher_metrics": {
            "robust_clean_validation_accuracy": float(robust_clean_acc),
            "robust_shifted_validation_accuracy": float(robust_shifted_acc),
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
