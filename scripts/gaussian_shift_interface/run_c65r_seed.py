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
from scripts.gaussian_shift_interface.run_c64r_seed import correction_for_k
from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import (
    FullSpanReplacedNet,
    accuracy,
    acts,
    compile_final_from_hierarchy,
    set_all_trainable,
    split_data,
    train_teacher,
)

FRESH_SEEDS = tuple(range(63400, 63416))
IMPLEMENTATION_VERIFICATION_SEED = 63300
GAUSSIAN_SIGMA = 0.04
CALIBRATION_SAMPLES = 192
FIT_UPDATES = 600
TOP_ADAPTATION_OFFSET = 655000
FINAL_INIT_OFFSET = 656000
FINAL_FIT_OFFSET = 657000


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
    Xc_shift = shifted_input(Xt[cal_t], seed + 650100)
    Xv_shift = shifted_input(Xv, seed + 650200)

    ac = acts(teacher, Xc_shift)
    av = acts(teacher, Xv_shift)
    a0c, a4c = ac[0], ac[-1]
    a0v, a4v = av[0], av[-1]
    denom = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    with torch.no_grad():
        baseline_c = base_hierarchy(a0c).detach()
        residual_c = a4c - baseline_c
    q = canonical_nested_qr(residual_c)

    # P0: no teacher correction and no top-boundary adaptation.
    p0_final, p0 = compile_final_from_hierarchy(
        copy.deepcopy(base_hierarchy),
        a0c,
        a0v,
        a4v,
        denom,
        seed + FINAL_INIT_OFFSET,
        seed + FINAL_FIT_OFFSET,
    )
    p0["final_replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(p0_final)), Xv_shift, yv
    )
    p0["dimension"] = 0
    p0["teacher_correction_applied"] = False
    p0["top_boundary_adaptation_skipped"] = True

    # P2: exactly the selected QR correction mechanism, with no P1 or other rescue.
    p2_hierarchy = copy.deepcopy(base_hierarchy)
    with torch.no_grad():
        correction = correction_for_k(residual_c, q, 2)
        p2_target = (baseline_c + correction).detach()
    adapt_anchored(
        p2_hierarchy,
        a0c,
        p2_target,
        FIT_UPDATES,
        seed + TOP_ADAPTATION_OFFSET,
    )
    set_all_trainable(p2_hierarchy, False)
    p2_final, p2 = compile_final_from_hierarchy(
        p2_hierarchy,
        a0c,
        a0v,
        a4v,
        denom,
        seed + FINAL_INIT_OFFSET,
        seed + FINAL_FIT_OFFSET,
    )
    p2["final_replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(p2_final)), Xv_shift, yv
    )
    p2["dimension"] = 2
    p2["teacher_correction_applied"] = True
    p2["top_boundary_adaptation_skipped"] = False

    required = [
        p0["final_replacement_val_acc"],
        p0["final_nmse_vs_original"],
        p2["final_replacement_val_acc"],
        p2["final_nmse_vs_original"],
    ]
    finite = all(math.isfinite(float(x)) for x in required)
    positive_nmse = (
        float(p0["final_nmse_vs_original"]) > 0.0
        and float(p2["final_nmse_vs_original"]) > 0.0
    )
    eligible = bool(budget["exact_4096_each_level"] and finite and positive_nmse)

    decision_row = {
        "seed": int(seed),
        "eligible": eligible,
        "p0_validation_accuracy": float(p0["final_replacement_val_acc"]),
        "p2_validation_accuracy": float(p2["final_replacement_val_acc"]),
        "p0_nmse": float(p0["final_nmse_vs_original"]),
        "p2_nmse": float(p2["final_nmse_vs_original"]),
    }

    return {
        "experiment": "C65R_P0_VS_P2_CONFIRMATION",
        "evidence_class": "PROSPECTIVE_CONFIRMATORY",
        "status": "FRESH_SEED_OUTCOME" if seed in FRESH_SEEDS else "IMPLEMENTATION_VERIFICATION_OUTCOME",
        "seed": int(seed),
        "eligible": eligible,
        "test_evaluated": False,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "protocol": {
            "sigma": GAUSSIAN_SIGMA,
            "calibration_samples": CALIBRATION_SAMPLES,
            "conditions": ["p0", "p2"],
            "p0_definition": "frozen base hierarchy compiled directly; no teacher residual correction and no top-boundary adaptation",
            "p2_top_adaptation_seed": seed + TOP_ADAPTATION_OFFSET,
            "final_init_seed": seed + FINAL_INIT_OFFSET,
            "final_fit_seed": seed + FINAL_FIT_OFFSET,
        },
        "provenance_hashes": {
            "calibration_indices_sha256": sha256_int_array(cal_idx),
            "shifted_calibration_tensor_sha256": sha256_tensor(Xc_shift),
            "shifted_validation_tensor_sha256": sha256_tensor(Xv_shift),
            "nested_q2_sha256": sha256_tensor(q[:, :2]),
        },
        "teacher_clean_validation_accuracy": accuracy(teacher, Xv, yv),
        "teacher_shifted_validation_accuracy": accuracy(teacher, Xv_shift, yv),
        "budget": budget,
        "conditions": {"p0": p0, "p2": p2},
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
