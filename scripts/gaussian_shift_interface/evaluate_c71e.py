#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = tuple(range(69400, 69416))
MINIMUM_ELIGIBLE = 8
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_SEED = 3326093687
TARGET_VALIDITY_MARGIN = -0.20
STAGE_REFERENCE_MARGIN = -0.05
STAGES = ("H64", "S64", "D64")


def percentile95(x: np.ndarray) -> list[float]:
    q = np.percentile(x, [2.5, 97.5])
    return [float(q[0]), float(q[1])]


def evaluate(payload: dict) -> dict:
    rows = list(payload.get("rows", []))
    by_seed = {int(r["seed"]): r for r in rows if "seed" in r}
    unexpected = sorted(set(by_seed) - set(EXPECTED_SEEDS))
    if unexpected:
        raise ValueError(f"unexpected seed rows: {unexpected}")
    missing = [
        s for s in EXPECTED_SEEDS
        if s not in by_seed or by_seed[s].get("failure") == "missing_seed_artifact"
    ]
    eligible = [
        by_seed[s] for s in EXPECTED_SEEDS
        if s in by_seed and bool(by_seed[s].get("eligible", False))
    ]

    out = {
        "experiment": "C71E_P64_PIPELINE_BOTTLENECK_LOCALIZATION",
        "evidence_class": "PROSPECTIVE_EXPLORATORY",
        "expected_seeds": list(EXPECTED_SEEDS),
        "attempted_rows_present": sum(
            1 for s in EXPECTED_SEEDS
            if s in by_seed and by_seed[s].get("failure") != "missing_seed_artifact"
        ),
        "eligible_count": len(eligible),
        "minimum_eligible": MINIMUM_ELIGIBLE,
        "missing_seed_rows": missing,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confirmatory_claim_allowed": False,
        "reduced_dimension_selection_allowed": False,
    }
    if len(eligible) < MINIMUM_ELIGIBLE:
        out["decision"] = "STOP_INSUFFICIENT_ELIGIBLE"
        return out

    robust_clean = np.asarray([
        float(r["robust_teacher_clean_validation_accuracy"]) for r in eligible
    ])
    robust_shift = np.asarray([
        float(r["robust_teacher_shifted_validation_accuracy"]) for r in eligible
    ])
    stage_acc = {
        "H64": np.asarray([float(r["h64_validation_accuracy"]) for r in eligible]),
        "S64": np.asarray([float(r["s64_validation_accuracy"]) for r in eligible]),
        "D64": np.asarray([float(r["d64_validation_accuracy"]) for r in eligible]),
    }
    stage_nmse = {
        "H64": np.asarray([float(r["h64_nmse_vs_teacher"]) for r in eligible]),
        "S64": np.asarray([float(r["s64_nmse_vs_teacher"]) for r in eligible]),
        "D64": np.asarray([float(r["d64_nmse_vs_teacher"]) for r in eligible]),
    }
    s64_nmse_h64 = np.asarray([float(r["s64_nmse_vs_h64"]) for r in eligible])
    h64_cal_nmse = np.asarray([float(r["h64_calibration_nmse_vs_teacher"]) for r in eligible])
    d64_cal_nmse = np.asarray([float(r["d64_calibration_nmse_vs_teacher"]) for r in eligible])
    full_basis_err = np.asarray([
        float(r["full_basis_relative_calibration_residual_sqerr"]) for r in eligible
    ])

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(
        0, len(eligible), size=(BOOTSTRAP_RESAMPLES, len(eligible)), endpoint=False
    )

    target_drop = robust_shift - robust_clean
    target_ci = percentile95(target_drop[idx].mean(axis=1))
    target_pass = bool(target_ci[0] > TARGET_VALIDITY_MARGIN)
    out["robust_target_validity"] = {
        "robust_clean_accuracy_mean": float(robust_clean.mean()),
        "robust_shifted_accuracy_mean": float(robust_shift.mean()),
        "mean_shift_minus_clean_pp": float(100.0 * target_drop.mean()),
        "bootstrap95_pp": [100.0 * x for x in target_ci],
        "margin_pp": 100.0 * TARGET_VALIDITY_MARGIN,
        "pass": target_pass,
    }

    stage_validity: dict[str, dict] = {}
    stage_pass: dict[str, bool] = {}
    for stage in STAGES:
        gap = stage_acc[stage] - robust_shift
        ci = percentile95(gap[idx].mean(axis=1))
        passed = bool(ci[0] > STAGE_REFERENCE_MARGIN)
        stage_pass[stage] = passed
        stage_validity[stage] = {
            "stage_accuracy_mean": float(stage_acc[stage].mean()),
            "robust_shifted_teacher_accuracy_mean": float(robust_shift.mean()),
            "mean_stage_minus_teacher_pp": float(100.0 * gap.mean()),
            "bootstrap95_pp": [100.0 * x for x in ci],
            "margin_pp": 100.0 * STAGE_REFERENCE_MARGIN,
            "pass": passed,
            "activation_nmse_vs_teacher_mean": float(stage_nmse[stage].mean()),
        }
    out["stage_reference_validity"] = stage_validity
    out["stage_validity_pattern"] = [stage_pass[s] for s in STAGES]

    s_minus_h = stage_acc["S64"] - stage_acc["H64"]
    d_minus_s = stage_acc["D64"] - stage_acc["S64"]
    s_minus_h_ci = percentile95(s_minus_h[idx].mean(axis=1))
    d_minus_s_ci = percentile95(d_minus_s[idx].mean(axis=1))
    out["descriptive_stage_diagnostics"] = {
        "s64_minus_h64_validation_accuracy": {
            "mean_pp": float(100.0 * s_minus_h.mean()),
            "bootstrap95_pp": [100.0 * x for x in s_minus_h_ci],
        },
        "d64_minus_s64_validation_accuracy": {
            "mean_pp": float(100.0 * d_minus_s.mean()),
            "bootstrap95_pp": [100.0 * x for x in d_minus_s_ci],
        },
        "h64_nmse_vs_teacher_mean": float(stage_nmse["H64"].mean()),
        "s64_nmse_vs_teacher_mean": float(stage_nmse["S64"].mean()),
        "d64_nmse_vs_teacher_mean": float(stage_nmse["D64"].mean()),
        "s64_nmse_vs_h64_mean": float(s64_nmse_h64.mean()),
        "h64_calibration_nmse_vs_teacher_mean": float(h64_cal_nmse.mean()),
        "d64_calibration_nmse_vs_teacher_mean": float(d64_cal_nmse.mean()),
        "max_full_basis_relative_calibration_residual_sqerr": float(full_basis_err.max()),
        "mean_full_basis_relative_calibration_residual_sqerr": float(full_basis_err.mean()),
    }

    h_valid = stage_pass["H64"]
    s_valid = stage_pass["S64"]
    d_valid = stage_pass["D64"]
    if not target_pass:
        decision = "STOP_ROBUST_TARGET_INVALID"
    elif h_valid and s_valid:
        decision = "STOP_C70_P64_FAILURE_NOT_REPRODUCED"
    elif (not h_valid) and s_valid:
        decision = "STOP_NONMONOTONIC_COMPILER_RESCUE"
    elif h_valid and (not s_valid):
        decision = "LOCALIZE_STANDARD_COMPILATION_LOSS"
    elif (not h_valid) and (not s_valid) and d_valid:
        decision = "LOCALIZE_HIERARCHY_ADAPTATION_LOSS"
    else:
        decision = "LOCALIZE_SHARED_MAPPING_OR_CALIBRATION_LIMIT"
    out["decision"] = decision

    out["interpretation_boundary"] = [
        "C71E is exploratory and localizes pipeline stages only; it selects no reduced interface dimension.",
        "H64 is the adapted recursive hierarchy before final compilation; S64 is the standard hierarchy-target compiler; D64 is a same-budget direct repaired-teacher-target compiler.",
        "A shared H64/D64 failure cannot distinguish 4096-parameter function capacity from calibration generalization or optimization and must not be over-localized.",
        "The -5 pp stage validity margin is a preregistered task-reference safeguard, not an equality criterion.",
        "This is exact C68E-repaired Residual-MLP evidence at Gaussian sigma=.36 and does not confirm the imported Residual CNN C59/C60 line."
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = evaluate(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
