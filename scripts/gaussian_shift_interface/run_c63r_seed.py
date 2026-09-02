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

FRESH_SEEDS = tuple(range(61400, 61416))
IMPLEMENTATION_VERIFICATION_SEED = 61300
GAUSSIAN_SIGMA = 0.04
CALIBRATION_SAMPLES = 192
FIT_UPDATES = 600
TOP_ADAPTATION_OFFSET = 635000
FINAL_INIT_OFFSET = 636000
FINAL_FIT_OFFSET = 637000


def evaluate_condition(
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
    hierarchy = copy.deepcopy(base_hierarchy)
    with torch.no_grad():
        baseline_c = base_hierarchy(a0c).detach()
        residual_c = a4c - baseline_c
        p = q[:, :k]
        correction = (residual_c @ p) @ p.T
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
    Xc_shift = shifted_input(Xt[cal_t], seed + 630100)
    Xv_shift = shifted_input(Xv, seed + 630200)

    ac = acts(teacher, Xc_shift)
    av = acts(teacher, Xv_shift)
    a0c, a4c = ac[0], ac[-1]
    a0v, a4v = av[0], av[-1]
    denom = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    with torch.no_grad():
        baseline_c = base_hierarchy(a0c).detach()
        residual_c = a4c - baseline_c
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

    p2 = evaluate_condition(
        teacher, base_hierarchy, q, 2, a0c, a4c, a0v, a4v,
        Xv_shift, yv, denom, seed
    )
    p4 = evaluate_condition(
        teacher, base_hierarchy, q, 4, a0c, a4c, a0v, a4v,
        Xv_shift, yv, denom, seed
    )

    required = [
        p2["final_replacement_val_acc"],
        p4["final_replacement_val_acc"],
        p2["final_nmse_vs_original"],
        p4["final_nmse_vs_original"],
    ]
    finite = all(math.isfinite(float(x)) for x in required)
    positive_nmse = (
        float(p2["final_nmse_vs_original"]) > 0.0
        and float(p4["final_nmse_vs_original"]) > 0.0
    )
    eligible = bool(budget["exact_4096_each_level"] and finite and positive_nmse)

    decision_row = {
        "seed": int(seed),
        "eligible": eligible,
        "p2_validation_accuracy": float(p2["final_replacement_val_acc"]),
        "p4_validation_accuracy": float(p4["final_replacement_val_acc"]),
        "p2_nmse": float(p2["final_nmse_vs_original"]),
        "p4_nmse": float(p4["final_nmse_vs_original"]),
        "frozen_nmse": float(frozen_metrics["final_nmse_vs_original"]),
    }

    return {
        "experiment": "C63R_QR_P2_VS_P4_CONFIRMATION",
        "evidence_class": "PROSPECTIVE_CONFIRMATORY",
        "status": "FRESH_SEED_OUTCOME" if seed in FRESH_SEEDS else "IMPLEMENTATION_VERIFICATION_OUTCOME",
        "seed": int(seed),
        "eligible": eligible,
        "test_evaluated": False,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "protocol": {
            "sigma": GAUSSIAN_SIGMA,
            "calibration_samples": CALIBRATION_SAMPLES,
            "dimensions": [2, 4],
            "top_adaptation_seed": seed + TOP_ADAPTATION_OFFSET,
            "final_init_seed": seed + FINAL_INIT_OFFSET,
            "final_fit_seed": seed + FINAL_FIT_OFFSET,
        },
        "provenance_hashes": {
            "calibration_indices_sha256": sha256_int_array(cal_idx),
            "shifted_calibration_tensor_sha256": sha256_tensor(Xc_shift),
            "shifted_validation_tensor_sha256": sha256_tensor(Xv_shift),
            "nested_q2_sha256": sha256_tensor(q[:, :2]),
            "nested_q4_sha256": sha256_tensor(q[:, :4]),
        },
        "teacher_clean_validation_accuracy": accuracy(teacher, Xv, yv),
        "teacher_shifted_validation_accuracy": accuracy(teacher, Xv_shift, yv),
        "budget": budget,
        "frozen": frozen_metrics,
        "p2": p2,
        "p4": p4,
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
