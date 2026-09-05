#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = tuple(range(65400, 65416))
MINIMUM_ELIGIBLE = 8
SIGMAS = (0.20, 0.28, 0.36, 0.44, 0.52, 0.60)
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_SEED = 3582014758
VALIDATION_MARGIN = -0.02
NMSE_RATIO_MARGIN = 1.25
TEACHER_DROP_MARGIN = -0.20
P2_TEACHER_GAP_MARGIN = -0.05


def sigma_key(sigma: float) -> str:
    return f"{sigma:.2f}"


def sigma_tag(sigma: float) -> str:
    return sigma_key(sigma).replace(".", "_")


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
    clean = np.asarray([float(r["teacher_clean_validation_accuracy"]) for r in rows])
    teacher_shift = np.asarray([
        float(r["sigma_conditions"][key]["mechanism"]["teacher_shifted_validation_accuracy"])
        for r in rows
    ])

    p0_p2_diff = p0_acc - p2_acc
    log_ratio = np.log(p0_nmse / p2_nmse)
    teacher_drop = teacher_shift - clean
    p2_teacher_gap = p2_acc - teacher_shift

    p0_p2_boot = p0_p2_diff[idx].mean(axis=1)
    ratio_boot = np.exp(log_ratio[idx].mean(axis=1))
    teacher_drop_boot = teacher_drop[idx].mean(axis=1)
    p2_teacher_gap_boot = p2_teacher_gap[idx].mean(axis=1)

    diff_ci = percentile95(p0_p2_boot)
    ratio_ci = percentile95(ratio_boot)
    teacher_drop_ci = percentile95(teacher_drop_boot)
    p2_teacher_gap_ci = percentile95(p2_teacher_gap_boot)

    val_pass = bool(diff_ci[0] > VALIDATION_MARGIN)
    nmse_pass = bool(ratio_ci[1] < NMSE_RATIO_MARGIN)
    teacher_valid = bool(teacher_drop_ci[0] > TEACHER_DROP_MARGIN)
    p2_valid = bool(p2_teacher_gap_ci[0] > P2_TEACHER_GAP_MARGIN)

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
            "mean_difference_fraction": float(p0_p2_diff.mean()),
            "mean_difference_pp": float(100.0 * p0_p2_diff.mean()),
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
        "p0_joint_pass": bool(val_pass and nmse_pass),
        "teacher_task_validity": {
            "clean_accuracy_mean": float(clean.mean()),
            "shifted_accuracy_mean": float(teacher_shift.mean()),
            "mean_drop_pp": float(100.0 * teacher_drop.mean()),
            "drop_bootstrap95_pp": [100.0 * x for x in teacher_drop_ci],
            "margin_pp": 100.0 * TEACHER_DROP_MARGIN,
            "pass": teacher_valid,
        },
        "p2_reference_validity": {
            "p2_accuracy_mean": float(p2_acc.mean()),
            "shifted_teacher_accuracy_mean": float(teacher_shift.mean()),
            "mean_gap_p2_minus_teacher_pp": float(100.0 * p2_teacher_gap.mean()),
            "gap_bootstrap95_pp": [100.0 * x for x in p2_teacher_gap_ci],
            "margin_pp": 100.0 * P2_TEACHER_GAP_MARGIN,
            "pass": p2_valid,
        },
        "validity_pass": bool(teacher_valid and p2_valid),
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
        "experiment": "C67E_EXTENDED_SHIFT_SEVERITY_FRONTIER_EXPLORATION",
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
        "selected_sigma_for_confirmation": None,
    }
    if len(eligible) < MINIMUM_ELIGIBLE:
        out["decision"] = "STOP_INSUFFICIENT_ELIGIBLE"
        return out

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(eligible), size=(BOOTSTRAP_RESAMPLES, len(eligible)), endpoint=False)
    curve = [per_sigma_summary(eligible, sigma, idx) for sigma in SIGMAS]
    out["severity_curve"] = {sigma_key(x["sigma"]): x for x in curve}
    p0_pattern = [bool(x["p0_joint_pass"]) for x in curve]
    validity_pattern = [bool(x["validity_pass"]) for x in curve]
    out["p0_joint_pass_pattern"] = p0_pattern
    out["validity_pass_pattern"] = validity_pattern

    if not validity_pattern[0]:
        out["decision"] = "STOP_C66R_ANCHOR_VALIDITY_INSTABILITY"
        out["validity_boundary_sigma"] = float(SIGMAS[0])
    elif not p0_pattern[0]:
        out["decision"] = "STOP_C66R_ANCHOR_P0_INSTABILITY"
        out["validity_boundary_sigma"] = None
    else:
        first_invalid = next((i for i, ok in enumerate(validity_pattern) if not ok), len(SIGMAS))
        out["validity_boundary_sigma"] = (
            float(SIGMAS[first_invalid]) if first_invalid < len(SIGMAS) else None
        )
        valid_p0 = p0_pattern[:first_invalid]
        if all(valid_p0):
            if first_invalid < len(SIGMAS):
                out["decision"] = f"STOP_VALIDITY_BOUNDARY_AT_SIGMA_{sigma_tag(SIGMAS[first_invalid])}"
            else:
                out["decision"] = "NO_P0_FAILURE_THROUGH_SIGMA_0_60"
        else:
            first_fail = valid_p0.index(False)
            recovery = any(valid_p0[j] for j in range(first_fail + 1, len(valid_p0)))
            if recovery:
                out["decision"] = "STOP_NONMONOTONIC_P0_FRONTIER"
            else:
                selected = SIGMAS[first_fail]
                out["decision"] = f"SELECT_SIGMA_{sigma_tag(selected)}_FOR_C68R"
                out["selected_sigma_for_confirmation"] = float(selected)

    out["interpretation_boundary"] = [
        "C67E is exploratory; any selected sigma is only a grid candidate and requires separate fresh C68R confirmation.",
        "A validity-boundary decision means teacher/reference degradation occurred before an interpretable P0 frontier was established; it must not be relabeled as P0 failure.",
        "The teacher-drop and P2-to-teacher margins are preregistered interpretability safeguards, not universal task-quality constants.",
        "The common standardized Gaussian direction across sigma is a variance-reduction device and does not make independent shifts equivalent.",
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
