#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gaussian_shift_interface.run_c61r_seed import (
    fixed_calibration_indices,
    sha256_int_array,
    sha256_tensor,
)
from scripts.gaussian_shift_interface.run_c68e_seed import state_dict_sha256, train_paired_teachers
from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import (
    FullSpanReplacedNet,
    TinyRes,
    accuracy,
    acts,
    count_params,
    fit_map,
    nmse,
    split_data,
)

FRESH_SEEDS = tuple(range(70400, 70416))
IMPLEMENTATION_VERIFICATION_SEED = 70300
GAUSSIAN_SIGMA = 0.36
BASE_CALIBRATION_SAMPLES = 192
LARGE_CALIBRATION_SAMPLES = 384
EXTENSION_INDEX_SEED = 20260906
BASE_CAL_EPS_OFFSET = 710100
EXT_CAL_EPS_OFFSET = 720101
VAL_EPS_OFFSET = 710200
FINAL_INIT_OFFSET = 716000
FINAL_FIT_OFFSET = 717000
FIT_UPDATES = 600
CELLS = {
    "N192_W32": (192, 32, 4096),
    "N384_W32": (384, 32, 4096),
    "N192_W64": (192, 64, 8192),
    "N384_W64": (384, 64, 8192),
}


def standard_normal_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(x.shape, generator=gen, dtype=x.dtype)


def shifted_input(x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x + GAUSSIAN_SIGMA * eps, 0.0, 1.0)


def nested_calibration_indices(n_train: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = fixed_calibration_indices(n_train)
    all_idx = np.arange(n_train, dtype=np.int64)
    complement = np.setdiff1d(all_idx, base, assume_unique=True)
    rng = np.random.default_rng(EXTENSION_INDEX_SEED)
    ext = np.sort(rng.choice(complement, size=BASE_CALIBRATION_SAMPLES, replace=False)).astype(np.int64)
    large = np.concatenate([base, ext]).astype(np.int64)
    if len(np.unique(large)) != LARGE_CALIBRATION_SAMPLES:
        raise AssertionError("nested calibration contains duplicate indices")
    if not np.array_equal(large[:BASE_CALIBRATION_SAMPLES], base):
        raise AssertionError("base calibration is not exact prefix of large calibration")
    return base, ext, large


def fit_direct(
    seed: int,
    width: int,
    a0c: torch.Tensor,
    a4c: torch.Tensor,
    a0v: torch.Tensor,
    a4v: torch.Tensor,
    teacher,
    xv_shift: torch.Tensor,
    yv: torch.Tensor,
    denom_teacher: float,
) -> tuple[dict, str]:
    model = TinyRes(64, width, seed + FINAL_INIT_OFFSET)
    init_hash = state_dict_sha256(model)
    params = count_params(model)
    model = fit_map(model, a0c, a4c.detach(), FIT_UPDATES, seed + FINAL_FIT_OFFSET)
    val_acc = accuracy(FullSpanReplacedNet(teacher, model), xv_shift, yv)
    with torch.no_grad():
        pred_c = model(a0c).detach()
        pred_v = model(a0v).detach()
    return {
        "width": int(width),
        "trainable_parameters": int(params),
        "validation_accuracy": float(val_acc),
        "activation_nmse_vs_teacher": float(nmse(pred_v, a4v, denom_teacher)),
        "calibration_nmse_vs_teacher": float(nmse(pred_c, a4c)),
        "final_state_sha256": state_dict_sha256(model),
    }, init_hash


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

    base_idx, ext_idx, large_idx = nested_calibration_indices(len(xt))
    base_t = torch.tensor(base_idx, dtype=torch.long)
    ext_t = torch.tensor(ext_idx, dtype=torch.long)

    x_base_clean = xt[base_t]
    x_ext_clean = xt[ext_t]
    base_eps = standard_normal_like(x_base_clean, seed + BASE_CAL_EPS_OFFSET)
    ext_eps = standard_normal_like(x_ext_clean, seed + EXT_CAL_EPS_OFFSET)
    val_eps = standard_normal_like(xv, seed + VAL_EPS_OFFSET)
    x_base_shift = shifted_input(x_base_clean, base_eps)
    x_ext_shift = shifted_input(x_ext_clean, ext_eps)
    x_large_shift = torch.cat([x_base_shift, x_ext_shift], dim=0)
    xv_shift = shifted_input(xv, val_eps)

    if not torch.equal(x_large_shift[:BASE_CALIBRATION_SAMPLES], x_base_shift):
        raise AssertionError("shifted base calibration is not exact prefix of shifted large calibration")

    base_acts = acts(teacher, x_base_shift)
    large_acts = acts(teacher, x_large_shift)
    val_acts = acts(teacher, xv_shift)
    a0_base, a4_base = base_acts[0], base_acts[-1]
    a0_large, a4_large = large_acts[0], large_acts[-1]
    a0v, a4v = val_acts[0], val_acts[-1]
    if not torch.equal(a0_large[:BASE_CALIBRATION_SAMPLES], a0_base):
        raise AssertionError("base input activations are not exact prefix")
    if not torch.equal(a4_large[:BASE_CALIBRATION_SAMPLES], a4_base):
        raise AssertionError("base target activations are not exact prefix")

    denom_teacher = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    robust_shift_acc = accuracy(teacher, xv_shift, yv)
    clean_teacher_shift_acc = accuracy(clean_teacher, xv_shift, yv)

    cells: dict[str, dict] = {}
    init_hashes: dict[str, str] = {}
    for name, (ncal, width, expected_params) in CELLS.items():
        if ncal == BASE_CALIBRATION_SAMPLES:
            a0c, a4c = a0_base, a4_base
        elif ncal == LARGE_CALIBRATION_SAMPLES:
            a0c, a4c = a0_large, a4_large
        else:
            raise AssertionError((name, ncal))
        rec, init_hash = fit_direct(
            seed, width, a0c, a4c, a0v, a4v,
            teacher, xv_shift, yv, denom_teacher,
        )
        rec["calibration_samples"] = int(ncal)
        if rec["trainable_parameters"] != expected_params:
            raise AssertionError((name, rec["trainable_parameters"], expected_params))
        cells[name] = rec
        init_hashes[name] = init_hash

    if init_hashes["N192_W32"] != init_hashes["N384_W32"]:
        raise AssertionError("width32 calibration-size cells do not share identical initialization")
    if init_hashes["N192_W64"] != init_hashes["N384_W64"]:
        raise AssertionError("width64 calibration-size cells do not share identical initialization")

    required = [robust_clean_acc, robust_shift_acc, clean_teacher_clean_acc, clean_teacher_shift_acc]
    for rec in cells.values():
        required.extend([
            rec["validation_accuracy"],
            rec["activation_nmse_vs_teacher"],
            rec["calibration_nmse_vs_teacher"],
        ])
    finite = all(math.isfinite(float(x)) for x in required)
    positive_nmse = all(float(rec["activation_nmse_vs_teacher"]) > 0.0 for rec in cells.values())
    eligible = bool(finite and positive_nmse)

    decision_row = {
        "seed": int(seed),
        "eligible": eligible,
        "robust_teacher_clean_validation_accuracy": float(robust_clean_acc),
        "robust_teacher_shifted_validation_accuracy": float(robust_shift_acc),
        "clean_teacher_clean_validation_accuracy": float(clean_teacher_clean_acc),
        "clean_teacher_shifted_validation_accuracy": float(clean_teacher_shift_acc),
        "cells": {
            name: {
                "validation_accuracy": float(rec["validation_accuracy"]),
                "activation_nmse_vs_teacher": float(rec["activation_nmse_vs_teacher"]),
                "calibration_nmse_vs_teacher": float(rec["calibration_nmse_vs_teacher"]),
                "calibration_samples": int(rec["calibration_samples"]),
                "trainable_parameters": int(rec["trainable_parameters"]),
            }
            for name, rec in cells.items()
        },
    }

    return {
        "experiment": "C72E_DIRECT_MAPPING_FACTORIAL_REPAIR",
        "evidence_class": "PROSPECTIVE_EXPLORATORY",
        "status": "FRESH_SEED_OUTCOME" if seed in FRESH_SEEDS else "IMPLEMENTATION_VERIFICATION_OUTCOME",
        "seed": int(seed),
        "eligible": eligible,
        "test_evaluated": False,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "protocol": {
            "sigma": GAUSSIAN_SIGMA,
            "base_calibration_samples": BASE_CALIBRATION_SAMPLES,
            "large_calibration_samples": LARGE_CALIBRATION_SAMPLES,
            "extension_index_seed": EXTENSION_INDEX_SEED,
            "base_calibration_epsilon_seed": seed + BASE_CAL_EPS_OFFSET,
            "extension_calibration_epsilon_seed": seed + EXT_CAL_EPS_OFFSET,
            "validation_epsilon_seed": seed + VAL_EPS_OFFSET,
            "final_init_seed": seed + FINAL_INIT_OFFSET,
            "final_fit_seed": seed + FINAL_FIT_OFFSET,
            "fit_updates": FIT_UPDATES,
        },
        "teacher_training_provenance": teacher_training_prov,
        "provenance_hashes": {
            "base_calibration_indices_sha256": sha256_int_array(base_idx),
            "extension_calibration_indices_sha256": sha256_int_array(ext_idx),
            "large_calibration_indices_sha256": sha256_int_array(large_idx),
            "base_calibration_standard_normal_sha256": sha256_tensor(base_eps),
            "extension_calibration_standard_normal_sha256": sha256_tensor(ext_eps),
            "validation_standard_normal_sha256": sha256_tensor(val_eps),
            "shifted_base_calibration_tensor_sha256": sha256_tensor(x_base_shift),
            "shifted_large_calibration_tensor_sha256": sha256_tensor(x_large_shift),
            "shifted_validation_tensor_sha256": sha256_tensor(xv_shift),
            "robust_teacher_final_state_sha256": state_dict_sha256(teacher),
            "paired_clean_teacher_final_state_sha256": state_dict_sha256(clean_teacher),
            "width32_initial_state_sha256": init_hashes["N192_W32"],
            "width64_initial_state_sha256": init_hashes["N192_W64"],
        },
        "nesting_invariants": {
            "base_indices_are_large_prefix": bool(np.array_equal(large_idx[:BASE_CALIBRATION_SAMPLES], base_idx)),
            "base_shifted_inputs_are_large_prefix": bool(torch.equal(x_large_shift[:BASE_CALIBRATION_SAMPLES], x_base_shift)),
            "base_input_activations_are_large_prefix": bool(torch.equal(a0_large[:BASE_CALIBRATION_SAMPLES], a0_base)),
            "base_target_activations_are_large_prefix": bool(torch.equal(a4_large[:BASE_CALIBRATION_SAMPLES], a4_base)),
            "width32_initialization_shared_across_calibration_sizes": init_hashes["N192_W32"] == init_hashes["N384_W32"],
            "width64_initialization_shared_across_calibration_sizes": init_hashes["N192_W64"] == init_hashes["N384_W64"],
        },
        "teacher_metrics": {
            "robust_clean_validation_accuracy": float(robust_clean_acc),
            "robust_shifted_validation_accuracy": float(robust_shift_acc),
            "clean_paired_clean_validation_accuracy": float(clean_teacher_clean_acc),
            "clean_paired_shifted_validation_accuracy": float(clean_teacher_shift_acc),
        },
        "cells": cells,
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
