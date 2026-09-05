#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = tuple(range(64400, 64416))
MINIMUM_ELIGIBLE = 8
SIGMAS = (0.04, 0.08, 0.12, 0.16, 0.20)
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_SEED = 4017607924
VALIDATION_MARGIN = -0.02
NMSE_RATIO_MARGIN = 1.25


def sigma_key(sigma: float) -> str:
    return f"{sigma:.2f}"


def percentile95(x: np.ndarray) -> list[float]:
    q = np.percentile(x, [2.5, 97.5])
    return [float(q[0]), float(q[1])]


def summary_stats(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def per_sigma_summary(rows: list[dict], sigma: float, idx: np.ndarray) -> dict:
    key = sigma_key(sigma)
    p0_acc = np.asarray([float(r["sigma_conditions"][key]["p0_validation_accuracy"]) for r in rows])
    p2_acc = np.asarray([float(r["sigma_conditions"][key]["p2_validation_accuracy"]) for r in rows])
    p0_nmse = np.asarray([float(r["sigma_conditions"][key]["p0_nmse"]) for r in rows])
    p2_nmse = np.asarray([float(r["sigma_conditions"][key]["p2_nmse"]) for r in rows])
    diff = p0_acc - p2_acc
    log_ratio = np.log(p0_nmse / p2_nmse)
    diff_boot = diff[idx].mean(axis=1)
    ratio_boot = np.exp(log_ratio[idx].mean(axis=1))
    diff_ci = percentile95(diff_boot)
    ratio_ci = percentile95(ratio_boot)
    val_pass = bool(diff_ci[0] > VALIDATION_MARGIN)
    nmse_pass = bool(ratio_ci[1] < NMSE_RATIO_MARGIN)

    clean = np.asarray([float(r["teacher_clean_validation_accuracy"]) for r in rows])
    teacher_shift = np.asarray([
        float(r["sigma_conditions"][key]["mechanism"]["teacher_shifted_validation_accuracy"])
        for r in rows
    ])
    teacher_drop = teacher_shift - clean
    teacher_drop_boot = teacher_drop[idx].mean(axis=1)

    mechanism_names = (
        "frozen_hierarchy_activation_nmse_vs_teacher",
        "p2_euclidean_capture_fraction",
        "p2_logit_l2_retained_ratio",
        "p2_fisher_retained_ratio",
    )
    mechanism = {}
    for name in mechanism_names:
        values = np.asarray([
            float(r["sigma_conditions"][key]["mechanism"][name]) for r in rows
        ])
        mechanism[name] = summary_stats(values)

    return {
        "sigma": float(sigma),
        "validation_noninferiority": {
            "mean_difference_fraction": float(diff.mean()),
            "mean_difference_pp": float(100.0 * diff.mean()),
            "bootstrap95_fraction": diff_ci,
            "bootstrap95_pp": [100.0 * x for x in diff_ci],
            "margin_fraction": VALIDATION_MARGIN,
            "margin_pp": 100.0 * VALIDATION_MARGIN,
            "pass": val_pass,
        },
        "nmse_ratio": {
            "geometric_mean_p0_over_p2": float(math.exp(float(log_ratio.mean()))),
            "bootstrap95": ratio_ci,
            "margin": NMSE_RATIO_MARGIN,
            "pass": nmse_pass,
        },
        "joint_pass": bool(val_pass and nmse_pass),
        "teacher_shift": {
            "clean_accuracy_mean": float(clean.mean()),
            "shifted_accuracy_mean": float(teacher_shift.mean()),
            "mean_drop_pp": float(100.0 * teacher_drop.mean()),
            "drop_bootstrap95_pp": [100.0 * x for x in percentile95(teacher_drop_boot)],
        },
        "mechanism_summary": mechanism,
    }


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
        "experiment": "C66R_SHIFT_SEVERITY_FRONTIER_EXPLORATION",
        "evidence_class": "PROSPECTIVE_EXPLORATORY",
        "expected_seeds": list(EXPECTED_SEEDS),
        "ordered_sigmas": list(SIGMAS),
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
    }
    if len(eligible) < MINIMUM_ELIGIBLE:
        out["decision"] = "STOP_INSUFFICIENT_ELIGIBLE"
        return out

    # Reuse exactly one bootstrap index matrix across sigma to preserve the locked dose pairing.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(eligible), size=(BOOTSTRAP_RESAMPLES, len(eligible)), endpoint=False)
    curve = [per_sigma_summary(eligible, sigma, idx) for sigma in SIGMAS]
    out["severity_curve"] = {sigma_key(x["sigma"]): x for x in curve}
    pattern = [bool(x["joint_pass"]) for x in curve]
    out["joint_pass_pattern"] = pattern

    if not pattern[0]:
        out["decision"] = "STOP_REPLICATION_INSTABILITY_AT_SIGMA_0_04"
        out["selected_sigma_for_confirmation"] = None
    elif all(pattern):
        out["decision"] = "NO_P0_FAILURE_THROUGH_SIGMA_0_20"
        out["selected_sigma_for_confirmation"] = None
    else:
        first_fail = pattern.index(False)
        nonmonotonic = any(pattern[j] for j in range(first_fail + 1, len(pattern)))
        if nonmonotonic:
            out["decision"] = "STOP_NONMONOTONIC_FRONTIER"
            out["selected_sigma_for_confirmation"] = None
        else:
            selected = SIGMAS[first_fail]
            tag = sigma_key(selected).replace(".", "_")
            out["decision"] = f"SELECT_SIGMA_{tag}_FOR_C67R"
            out["selected_sigma_for_confirmation"] = float(selected)

    out["interpretation_boundary"] = [
        "C66R is exploratory; a selected sigma is only a grid candidate and requires a separate fresh confirmation.",
        "The common standardized Gaussian direction across sigma is a variance-reduction device for the dose-response experiment; it does not make independent shifts equivalent.",
        "The selected grid value is not an estimate of an exact continuous critical sigma.",
        "Teacher shifted accuracy and task-weighted residual diagnostics are descriptive/mechanism-generating and do not alter the locked frontier decision.",
        "This is Residual-MLP evidence and does not confirm the imported Residual CNN C59/C60 line."
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
