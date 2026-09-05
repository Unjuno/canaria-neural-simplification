#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = tuple(range(68400, 68416))
MINIMUM_ELIGIBLE = 8
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_SEED = 477816929
TARGET_VALIDITY_MARGIN = -0.20
P64_REFERENCE_MARGIN = -0.05


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
        "experiment": "C70E_P64_REFERENCE_REPAIR_EXPLORATION",
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
    }
    if len(eligible) < MINIMUM_ELIGIBLE:
        out["decision"] = "STOP_INSUFFICIENT_ELIGIBLE"
        return out

    robust_clean = np.asarray([float(r["robust_teacher_clean_validation_accuracy"]) for r in eligible])
    robust_shift = np.asarray([float(r["robust_teacher_shifted_validation_accuracy"]) for r in eligible])
    p32_acc = np.asarray([float(r["p32_validation_accuracy"]) for r in eligible])
    p64_acc = np.asarray([float(r["p64_validation_accuracy"]) for r in eligible])
    p32_nmse = np.asarray([float(r["p32_nmse"]) for r in eligible])
    p64_nmse = np.asarray([float(r["p64_nmse"]) for r in eligible])
    full_basis_err = np.asarray([float(r["full_basis_relative_calibration_residual_sqerr"]) for r in eligible])
    if np.any(p32_nmse <= 0.0) or np.any(p64_nmse <= 0.0):
        raise ValueError("non-positive NMSE in eligible rows")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(eligible), size=(BOOTSTRAP_RESAMPLES, len(eligible)), endpoint=False)

    target_drop = robust_shift - robust_clean
    p64_gap = p64_acc - robust_shift
    p32_gap = p32_acc - robust_shift
    p64_p32_acc = p64_acc - p32_acc
    log_nmse_ratio = np.log(p64_nmse / p32_nmse)

    target_ci = percentile95(target_drop[idx].mean(axis=1))
    p64_ci = percentile95(p64_gap[idx].mean(axis=1))
    p32_ci = percentile95(p32_gap[idx].mean(axis=1))
    p64_p32_ci = percentile95(p64_p32_acc[idx].mean(axis=1))
    ratio_ci = percentile95(np.exp(log_nmse_ratio[idx].mean(axis=1)))

    target_pass = bool(target_ci[0] > TARGET_VALIDITY_MARGIN)
    p64_pass = bool(p64_ci[0] > P64_REFERENCE_MARGIN)

    out["robust_target_validity"] = {
        "robust_clean_accuracy_mean": float(robust_clean.mean()),
        "robust_shifted_accuracy_mean": float(robust_shift.mean()),
        "mean_shift_minus_clean_pp": float(100.0 * target_drop.mean()),
        "bootstrap95_pp": [100.0 * x for x in target_ci],
        "margin_pp": 100.0 * TARGET_VALIDITY_MARGIN,
        "pass": target_pass,
    }
    out["p64_reference_validity"] = {
        "p64_accuracy_mean": float(p64_acc.mean()),
        "robust_shifted_teacher_accuracy_mean": float(robust_shift.mean()),
        "mean_p64_minus_teacher_pp": float(100.0 * p64_gap.mean()),
        "bootstrap95_pp": [100.0 * x for x in p64_ci],
        "margin_pp": 100.0 * P64_REFERENCE_MARGIN,
        "pass": p64_pass,
    }
    out["p32_reference_descriptive"] = {
        "p32_accuracy_mean": float(p32_acc.mean()),
        "mean_p32_minus_teacher_pp": float(100.0 * p32_gap.mean()),
        "bootstrap95_pp": [100.0 * x for x in p32_ci],
    }
    out["p64_vs_p32_descriptive"] = {
        "mean_validation_accuracy_difference_p64_minus_p32_pp": float(100.0 * p64_p32_acc.mean()),
        "validation_bootstrap95_pp": [100.0 * x for x in p64_p32_ci],
        "nmse_geometric_mean_ratio_p64_over_p32": float(math.exp(float(log_nmse_ratio.mean()))),
        "nmse_ratio_bootstrap95": ratio_ci,
    }
    out["full_basis_implementation_descriptive"] = {
        "max_relative_calibration_residual_sqerr_among_eligible": float(full_basis_err.max()),
        "mean_relative_calibration_residual_sqerr": float(full_basis_err.mean()),
    }

    if not target_pass:
        out["decision"] = "STOP_ROBUST_TARGET_INVALID"
    elif p64_pass:
        out["decision"] = "ADVANCE_P64_REFERENCE_TO_C71E"
    else:
        out["decision"] = "STOP_P64_REFERENCE_INVALID"

    out["interpretation_boundary"] = [
        "C70E is exploratory and tests reference repair only; it performs no reduced-dimension candidate selection.",
        "P64 uses all 64 columns of the canonical calibration QR basis; this is full-basis calibration correction, not a universal exact model representation.",
        "Passing the calibration reconstruction invariant is implementation equivalence only and does not guarantee final reference validity after adaptation/compilation.",
        "P64-versus-P32 accuracy and NMSE differences are descriptive diagnostics and are not additional decision gates.",
        "A P64 advance only authorizes a separately locked fresh C71E reduced-dimension frontier experiment."
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
