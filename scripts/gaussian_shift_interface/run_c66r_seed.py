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
)
from scripts.gaussian_shift_interface.run_c64r_seed import (
    correction_for_k,
    euclidean_capture,
    fisher_retained_ratio,
    logit_l2_retained_ratio,
)
from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import (
    FullSpanReplacedNet,
    accuracy,
    acts,
    compile_final_from_hierarchy,
    nmse,
    set_all_trainable,
    split_data,
    train_teacher,
)

FRESH_SEEDS = tuple(range(64400, 64416))
IMPLEMENTATION_VERIFICATION_SEED = 64300
SIGMAS = (0.04, 0.08, 0.12, 0.16, 0.20)
CALIBRATION_SAMPLES = 192
FIT_UPDATES = 600
CAL_EPS_OFFSET = 660100
VAL_EPS_OFFSET = 660200
TOP_ADAPTATION_OFFSET = 665000
FINAL_INIT_OFFSET = 666000
FINAL_FIT_OFFSET = 667000


def sigma_key(sigma: float) -> str:
    return f"{sigma:.2f}"


def standard_normal_like(x: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(x.shape, generator=gen, dtype=x.dtype)


def apply_scaled_gaussian(x: torch.Tensor, epsilon: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.clamp(x + float(sigma) * epsilon, 0.0, 1.0)


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
    teacher_clean_acc = accuracy(teacher, Xv, yv)
    a0t, a1t, a2t, a3t, a4t = acts(teacher, Xt)
    base_hierarchy, budget = build_base_hierarchy(seed, a0t, a1t, a2t, a3t, a4t)

    cal_idx = fixed_calibration_indices(len(Xt))
    cal_t = torch.tensor(cal_idx, dtype=torch.long)
    Xc_clean = Xt[cal_t]
    cal_eps = standard_normal_like(Xc_clean, seed + CAL_EPS_OFFSET)
    val_eps = standard_normal_like(Xv, seed + VAL_EPS_OFFSET)
    head_weight = teacher.head.weight.detach()

    sigma_results: dict[str, dict] = {}
    provenance: dict[str, dict] = {}
    all_required: list[float] = []

    for sigma in SIGMAS:
        skey = sigma_key(sigma)
        Xc_shift = apply_scaled_gaussian(Xc_clean, cal_eps, sigma)
        Xv_shift = apply_scaled_gaussian(Xv, val_eps, sigma)
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
            teacher_prob_v = F.softmax(teacher(Xv_shift), dim=1)
        q = canonical_nested_qr(residual_c)

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

        p2_hierarchy = copy.deepcopy(base_hierarchy)
        with torch.no_grad():
            correction_c = correction_for_k(residual_c, q, 2)
            p2_target = (baseline_c + correction_c).detach()
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

        with torch.no_grad():
            correction_v = correction_for_k(residual_v, q, 2)
        mechanism = {
            "teacher_shifted_validation_accuracy": accuracy(teacher, Xv_shift, yv),
            "teacher_accuracy_drop_from_clean_pp": 100.0 * (
                accuracy(teacher, Xv_shift, yv) - teacher_clean_acc
            ),
            "frozen_hierarchy_activation_nmse_vs_teacher": nmse(baseline_v, a4v, denom),
            "p2_euclidean_capture_fraction": euclidean_capture(residual_v, correction_v),
            "p2_logit_l2_retained_ratio": logit_l2_retained_ratio(
                residual_v, correction_v, head_weight
            ),
            "p2_fisher_retained_ratio": fisher_retained_ratio(
                residual_v, correction_v, head_weight, teacher_prob_v
            ),
        }

        sigma_results[skey] = {
            "sigma": float(sigma),
            "p0": p0,
            "p2": p2,
            "mechanism": mechanism,
        }
        provenance[skey] = {
            "shifted_calibration_tensor_sha256": sha256_tensor(Xc_shift),
            "shifted_validation_tensor_sha256": sha256_tensor(Xv_shift),
            "nested_q2_sha256": sha256_tensor(q[:, :2]),
        }
        all_required.extend([
            p0["final_replacement_val_acc"],
            p0["final_nmse_vs_original"],
            p2["final_replacement_val_acc"],
            p2["final_nmse_vs_original"],
            mechanism["teacher_shifted_validation_accuracy"],
            mechanism["teacher_accuracy_drop_from_clean_pp"],
            mechanism["frozen_hierarchy_activation_nmse_vs_teacher"],
            mechanism["p2_euclidean_capture_fraction"],
            mechanism["p2_logit_l2_retained_ratio"],
            mechanism["p2_fisher_retained_ratio"],
        ])

    finite = all(math.isfinite(float(x)) for x in all_required)
    positive_nmse = all(
        float(sigma_results[sigma_key(s)][p]["final_nmse_vs_original"]) > 0.0
        for s in SIGMAS for p in ("p0", "p2")
    )
    nonnegative_mechanism = all(
        sigma_results[sigma_key(s)]["mechanism"][name] >= 0.0
        for s in SIGMAS
        for name in (
            "frozen_hierarchy_activation_nmse_vs_teacher",
            "p2_euclidean_capture_fraction",
            "p2_logit_l2_retained_ratio",
            "p2_fisher_retained_ratio",
        )
    )
    eligible = bool(
        budget["exact_4096_each_level"] and finite and positive_nmse and nonnegative_mechanism
    )

    decision_row = {
        "seed": int(seed),
        "eligible": eligible,
        "teacher_clean_validation_accuracy": float(teacher_clean_acc),
        "sigma_conditions": {
            skey: {
                "sigma": rec["sigma"],
                "p0_validation_accuracy": float(rec["p0"]["final_replacement_val_acc"]),
                "p2_validation_accuracy": float(rec["p2"]["final_replacement_val_acc"]),
                "p0_nmse": float(rec["p0"]["final_nmse_vs_original"]),
                "p2_nmse": float(rec["p2"]["final_nmse_vs_original"]),
                "mechanism": {k: float(v) for k, v in rec["mechanism"].items()},
            }
            for skey, rec in sigma_results.items()
        },
    }

    return {
        "experiment": "C66R_SHIFT_SEVERITY_FRONTIER_EXPLORATION",
        "evidence_class": "PROSPECTIVE_EXPLORATORY",
        "status": "FRESH_SEED_OUTCOME" if seed in FRESH_SEEDS else "IMPLEMENTATION_VERIFICATION_OUTCOME",
        "seed": int(seed),
        "eligible": eligible,
        "test_evaluated": False,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "protocol": {
            "sigmas": list(SIGMAS),
            "calibration_samples": CALIBRATION_SAMPLES,
            "common_calibration_epsilon_seed": seed + CAL_EPS_OFFSET,
            "common_validation_epsilon_seed": seed + VAL_EPS_OFFSET,
            "p2_top_adaptation_seed": seed + TOP_ADAPTATION_OFFSET,
            "final_init_seed": seed + FINAL_INIT_OFFSET,
            "final_fit_seed": seed + FINAL_FIT_OFFSET,
            "conditions": ["p0", "p2"],
        },
        "provenance_hashes": {
            "calibration_indices_sha256": sha256_int_array(cal_idx),
            "calibration_standard_normal_sha256": sha256_tensor(cal_eps),
            "validation_standard_normal_sha256": sha256_tensor(val_eps),
            "teacher_head_weight_sha256": sha256_tensor(head_weight),
            "per_sigma": provenance,
        },
        "teacher_clean_validation_accuracy": float(teacher_clean_acc),
        "budget": budget,
        "sigma_results": sigma_results,
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
