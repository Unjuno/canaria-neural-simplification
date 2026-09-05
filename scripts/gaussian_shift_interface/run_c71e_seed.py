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
    TinyRes,
    accuracy,
    acts,
    compile_final_from_hierarchy,
    fit_map,
    nmse,
    set_all_trainable,
    split_data,
)

FRESH_SEEDS = tuple(range(69400, 69416))
IMPLEMENTATION_VERIFICATION_SEED = 69300
GAUSSIAN_SIGMA = 0.36
CALIBRATION_SAMPLES = 192
FIT_UPDATES = 600
CAL_EPS_OFFSET = 710100
VAL_EPS_OFFSET = 710200
TOP_ADAPTATION_OFFSET = 715000
FINAL_INIT_OFFSET = 716000
FINAL_FIT_OFFSET = 717000
FULL_BASIS_REL_SQERR_MAX = 1e-10


def standard_normal_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(x.shape, generator=gen, dtype=x.dtype)


def shifted_input(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x + GAUSSIAN_SIGMA * eps, 0.0, 1.0)


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
    clean_teacher_clean_acc = accuracy(clean_teacher, xv, yv)

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
    denom_teacher = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    with torch.no_grad():
        baseline_c = base_hierarchy(a0c).detach()
        residual_c = a4c - baseline_c
    q = canonical_nested_qr(residual_c)
    if q.shape[1] < 64:
        raise AssertionError(("insufficient QR dimension", tuple(q.shape)))
    with torch.no_grad():
        p64_correction = correction_for_k(residual_c, q, 64)
        p64_target = (baseline_c + p64_correction).detach()
        full_basis_rel_sqerr = float(
            ((residual_c - p64_correction) ** 2).sum()
            / (((residual_c) ** 2).sum() + 1e-30)
        )
    full_basis_ok = bool(full_basis_rel_sqerr <= FULL_BASIS_REL_SQERR_MAX)

    # H64: P64-adapted recursive hierarchy before final compilation.
    h64 = copy.deepcopy(base_hierarchy)
    adapt_anchored(
        h64,
        a0c,
        p64_target,
        FIT_UPDATES,
        seed + TOP_ADAPTATION_OFFSET,
    )
    set_all_trainable(h64, False)
    with torch.no_grad():
        h64_cal = h64(a0c).detach()
        h64_val = h64(a0v).detach()
    h64_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(h64)), xv_shift, yv)
    h64_nmse_teacher = nmse(h64_val, a4v, denom_teacher)

    # S64: standard compiler fitted to H64 calibration targets.
    s64_final, s64_metrics = compile_final_from_hierarchy(
        copy.deepcopy(h64),
        a0c,
        a0v,
        a4v,
        denom_teacher,
        seed + FINAL_INIT_OFFSET,
        seed + FINAL_FIT_OFFSET,
    )
    s64_acc = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(s64_final)), xv_shift, yv
    )
    with torch.no_grad():
        s64_val = s64_final(a0v).detach()
    s64_nmse_teacher = nmse(s64_val, a4v, denom_teacher)

    # D64: identical 4096-parameter compiler, same init + minibatch stream as S64,
    # but fitted directly to repaired-teacher shifted-calibration final activations.
    d64_final = TinyRes(64, 32, seed + FINAL_INIT_OFFSET)
    d64_final = fit_map(
        d64_final,
        a0c,
        a4c.detach(),
        FIT_UPDATES,
        seed + FINAL_FIT_OFFSET,
    )
    d64_acc = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(d64_final)), xv_shift, yv
    )
    with torch.no_grad():
        d64_cal = d64_final(a0c).detach()
        d64_val = d64_final(a0v).detach()
    d64_nmse_teacher = nmse(d64_val, a4v, denom_teacher)

    robust_shifted_acc = accuracy(teacher, xv_shift, yv)
    clean_teacher_shifted_acc = accuracy(clean_teacher, xv_shift, yv)

    # Diagnostics that distinguish stage losses without becoming decision gates.
    denom_h64 = float(((h64_val - h64_val.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    s64_nmse_h64 = nmse(s64_val, h64_val, denom_h64)
    h64_cal_nmse_teacher = nmse(h64_cal, a4c)
    d64_cal_nmse_teacher = nmse(d64_cal, a4c)

    required = [
        robust_clean_acc,
        robust_shifted_acc,
        clean_teacher_clean_acc,
        clean_teacher_shifted_acc,
        full_basis_rel_sqerr,
        h64_acc,
        h64_nmse_teacher,
        s64_acc,
        s64_nmse_teacher,
        s64_nmse_h64,
        d64_acc,
        d64_nmse_teacher,
        h64_cal_nmse_teacher,
        d64_cal_nmse_teacher,
    ]
    finite = all(math.isfinite(float(x)) for x in required)
    positive_nmse = bool(s64_nmse_teacher > 0.0 and d64_nmse_teacher > 0.0)
    eligible = bool(
        budget["exact_4096_each_level"]
        and finite
        and positive_nmse
        and full_basis_ok
    )

    decision_row = {
        "seed": int(seed),
        "eligible": eligible,
        "full_basis_relative_calibration_residual_sqerr": full_basis_rel_sqerr,
        "robust_teacher_clean_validation_accuracy": float(robust_clean_acc),
        "robust_teacher_shifted_validation_accuracy": float(robust_shifted_acc),
        "clean_teacher_clean_validation_accuracy": float(clean_teacher_clean_acc),
        "clean_teacher_shifted_validation_accuracy": float(clean_teacher_shifted_acc),
        "h64_validation_accuracy": float(h64_acc),
        "s64_validation_accuracy": float(s64_acc),
        "d64_validation_accuracy": float(d64_acc),
        "h64_nmse_vs_teacher": float(h64_nmse_teacher),
        "s64_nmse_vs_teacher": float(s64_nmse_teacher),
        "d64_nmse_vs_teacher": float(d64_nmse_teacher),
        "s64_nmse_vs_h64": float(s64_nmse_h64),
        "h64_calibration_nmse_vs_teacher": float(h64_cal_nmse_teacher),
        "d64_calibration_nmse_vs_teacher": float(d64_cal_nmse_teacher),
    }

    return {
        "experiment": "C71E_P64_PIPELINE_BOTTLENECK_LOCALIZATION",
        "evidence_class": "PROSPECTIVE_EXPLORATORY",
        "status": "FRESH_SEED_OUTCOME" if seed in FRESH_SEEDS else "IMPLEMENTATION_VERIFICATION_OUTCOME",
        "seed": int(seed),
        "eligible": eligible,
        "test_evaluated": False,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "protocol": {
            "sigma": GAUSSIAN_SIGMA,
            "calibration_samples": CALIBRATION_SAMPLES,
            "calibration_epsilon_seed": seed + CAL_EPS_OFFSET,
            "validation_epsilon_seed": seed + VAL_EPS_OFFSET,
            "p64_top_adaptation_seed": seed + TOP_ADAPTATION_OFFSET,
            "standard_and_direct_final_init_seed": seed + FINAL_INIT_OFFSET,
            "standard_and_direct_final_fit_seed": seed + FINAL_FIT_OFFSET,
            "full_basis_relative_sqerr_threshold": FULL_BASIS_REL_SQERR_MAX,
            "stages": ["H64", "S64", "D64"],
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
            "nested_q64_sha256": sha256_tensor(q[:, :64]),
            "p64_calibration_target_sha256": sha256_tensor(p64_target),
        },
        "budget": budget,
        "implementation_invariant": {
            "full_basis_relative_calibration_residual_sqerr": full_basis_rel_sqerr,
            "threshold": FULL_BASIS_REL_SQERR_MAX,
            "pass": full_basis_ok,
        },
        "teacher_metrics": {
            "robust_clean_validation_accuracy": float(robust_clean_acc),
            "robust_shifted_validation_accuracy": float(robust_shifted_acc),
            "clean_paired_clean_validation_accuracy": float(clean_teacher_clean_acc),
            "clean_paired_shifted_validation_accuracy": float(clean_teacher_shifted_acc),
        },
        "stage_metrics": {
            "H64": {
                "validation_accuracy": float(h64_acc),
                "nmse_vs_teacher": float(h64_nmse_teacher),
                "calibration_nmse_vs_teacher": float(h64_cal_nmse_teacher),
            },
            "S64": {
                "validation_accuracy": float(s64_acc),
                "nmse_vs_teacher": float(s64_nmse_teacher),
                "nmse_vs_h64": float(s64_nmse_h64),
                "compiler_reported_final_nmse_vs_hierarchy": float(s64_metrics["final_nmse_vs_hierarchy"]),
            },
            "D64": {
                "validation_accuracy": float(d64_acc),
                "nmse_vs_teacher": float(d64_nmse_teacher),
                "calibration_nmse_vs_teacher": float(d64_cal_nmse_teacher),
            },
        },
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
