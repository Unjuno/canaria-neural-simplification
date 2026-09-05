#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = tuple(range(67400, 67416))
MINIMUM_ELIGIBLE = 8
CANDIDATE_DIMENSIONS = (0, 1, 2, 4, 8, 16)
REFERENCE_DIMENSION = 32
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_SEED = 491338359
TARGET_VALIDITY_MARGIN = -0.20
REFERENCE_VALIDITY_MARGIN = -0.05
VALIDATION_MARGIN = -0.02
NMSE_RATIO_MARGIN = 1.25


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
        "experiment": "C69E_ROBUST_TEACHER_INTERFACE_FRONTIER_EXPLORATION",
        "evidence_class": "PROSPECTIVE_EXPLORATORY",
        "expected_seeds": list(EXPECTED_SEEDS),
        "candidate_dimensions": list(CANDIDATE_DIMENSIONS),
        "reference_dimension": REFERENCE_DIMENSION,
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
        "selected_candidate_dimension": None,
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
    clean_clean = np.asarray([
        float(r["clean_teacher_clean_validation_accuracy"]) for r in eligible
    ])
    clean_shift = np.asarray([
        float(r["clean_teacher_shifted_validation_accuracy"]) for r in eligible
    ])
    p32_acc = np.asarray([
        float(r["conditions"]["p32"]["validation_accuracy"]) for r in eligible
    ])
    p32_nmse = np.asarray([
        float(r["conditions"]["p32"]["nmse"]) for r in eligible
    ])
    if np.any(p32_nmse <= 0.0):
        raise ValueError("non-positive P32 NMSE in eligible rows")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(
        0, len(eligible), size=(BOOTSTRAP_RESAMPLES, len(eligible)), endpoint=False
    )

    target_drop = robust_shift - robust_clean
    target_boot = target_drop[idx].mean(axis=1)
    target_ci = percentile95(target_boot)
    target_pass = bool(target_ci[0] > TARGET_VALIDITY_MARGIN)

    reference_gap = p32_acc - robust_shift
    reference_boot = reference_gap[idx].mean(axis=1)
    reference_ci = percentile95(reference_boot)
    reference_pass = bool(reference_ci[0] > REFERENCE_VALIDITY_MARGIN)

    out["robust_target_validity"] = {
        "robust_clean_accuracy_mean": float(robust_clean.mean()),
        "robust_shifted_accuracy_mean": float(robust_shift.mean()),
        "mean_shift_minus_clean_pp": float(100.0 * target_drop.mean()),
        "bootstrap95_pp": [100.0 * x for x in target_ci],
        "margin_pp": 100.0 * TARGET_VALIDITY_MARGIN,
        "pass": target_pass,
    }
    out["p32_reference_validity"] = {
        "p32_accuracy_mean": float(p32_acc.mean()),
        "robust_shifted_teacher_accuracy_mean": float(robust_shift.mean()),
        "mean_p32_minus_teacher_pp": float(100.0 * reference_gap.mean()),
        "bootstrap95_pp": [100.0 * x for x in reference_ci],
        "margin_pp": 100.0 * REFERENCE_VALIDITY_MARGIN,
        "pass": reference_pass,
    }
    baseline_drop = clean_shift - clean_clean
    out["paired_clean_teacher_descriptive"] = {
        "clean_accuracy_mean": float(clean_clean.mean()),
        "shifted_accuracy_mean": float(clean_shift.mean()),
        "mean_shift_minus_clean_pp": float(100.0 * baseline_drop.mean()),
    }

    candidate_curve: dict[str, dict] = {}
    pass_pattern: list[bool] = []
    for k in CANDIDATE_DIMENSIONS:
        key = f"p{k}"
        acc = np.asarray([
            float(r["conditions"][key]["validation_accuracy"]) for r in eligible
        ])
        nmse = np.asarray([
            float(r["conditions"][key]["nmse"]) for r in eligible
        ])
        if np.any(nmse <= 0.0):
            raise ValueError(f"non-positive {key} NMSE in eligible rows")
        diff = acc - p32_acc
        log_ratio = np.log(nmse / p32_nmse)
        diff_boot = diff[idx].mean(axis=1)
        ratio_boot = np.exp(log_ratio[idx].mean(axis=1))
        diff_ci = percentile95(diff_boot)
        ratio_ci = percentile95(ratio_boot)
        val_pass = bool(diff_ci[0] > VALIDATION_MARGIN)
        nmse_pass = bool(ratio_ci[1] < NMSE_RATIO_MARGIN)
        joint = bool(val_pass and nmse_pass)
        pass_pattern.append(joint)
        candidate_curve[key] = {
            "dimension": k,
            "validation_noninferiority": {
                "mean_difference_candidate_minus_p32_fraction": float(diff.mean()),
                "mean_difference_candidate_minus_p32_pp": float(100.0 * diff.mean()),
                "bootstrap95_fraction": diff_ci,
                "bootstrap95_pp": [100.0 * x for x in diff_ci],
                "margin_fraction": VALIDATION_MARGIN,
                "margin_pp": 100.0 * VALIDATION_MARGIN,
                "pass": val_pass,
            },
            "nmse_ratio": {
                "geometric_mean_candidate_over_p32": float(math.exp(float(log_ratio.mean()))),
                "bootstrap95": ratio_ci,
                "margin": NMSE_RATIO_MARGIN,
                "pass": nmse_pass,
            },
            "joint_pass": joint,
        }

    out["candidate_curve"] = candidate_curve
    out["candidate_joint_pass_pattern"] = pass_pattern

    if not target_pass:
        out["decision"] = "STOP_ROBUST_TARGET_INVALID"
    elif not reference_pass:
        out["decision"] = "STOP_P32_REFERENCE_INVALID"
    else:
        first_true = next((i for i, ok in enumerate(pass_pattern) if ok), None)
        if first_true is None:
            out["decision"] = "NO_REDUCED_CANDIDATE_THROUGH_P16"
        elif any(not ok for ok in pass_pattern[first_true:]):
            out["decision"] = "STOP_NONMONOTONIC_DIMENSION_FRONTIER"
        else:
            selected = CANDIDATE_DIMENSIONS[first_true]
            out["selected_candidate_dimension"] = selected
            out["decision"] = f"ADVANCE_P{selected}_TO_C70R"

    out["interpretation_boundary"] = [
        "C69E is exploratory; any selected dimension requires a separate fresh C70R confirmation.",
        "P32 is a locked strong reference but is not claimed to be exact/full residual reconstruction.",
        "A robust-target or P32-reference validity failure stops candidate interpretation before dimension selection.",
        "A nonmonotonic dimension pass pattern is not converted into a convenient smallest passing dimension.",
        "This result belongs to the exact C68E repaired-teacher Residual-MLP at Gaussian sigma=.36 and does not confirm the imported Residual CNN C59/C60 line."
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
