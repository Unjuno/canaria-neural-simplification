#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = tuple(range(70400, 70416))
MINIMUM_ELIGIBLE = 8
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_SEED = 4236432130
TARGET_VALIDITY_MARGIN = -0.20
CELL_REFERENCE_MARGIN = -0.05
CELLS = ("N192_W32", "N384_W32", "N192_W64", "N384_W64")


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
        "experiment": "C72E_DIRECT_MAPPING_FACTORIAL_REPAIR",
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
    cell_acc = {
        c: np.asarray([float(r["cells"][c]["validation_accuracy"]) for r in eligible])
        for c in CELLS
    }
    cell_nmse = {
        c: np.asarray([float(r["cells"][c]["activation_nmse_vs_teacher"]) for r in eligible])
        for c in CELLS
    }
    cell_cal_nmse = {
        c: np.asarray([float(r["cells"][c]["calibration_nmse_vs_teacher"]) for r in eligible])
        for c in CELLS
    }

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

    validity = {}
    pass_map = {}
    for c in CELLS:
        gap = cell_acc[c] - robust_shift
        ci = percentile95(gap[idx].mean(axis=1))
        passed = bool(ci[0] > CELL_REFERENCE_MARGIN)
        pass_map[c] = passed
        validity[c] = {
            "cell_accuracy_mean": float(cell_acc[c].mean()),
            "robust_shifted_teacher_accuracy_mean": float(robust_shift.mean()),
            "mean_cell_minus_teacher_pp": float(100.0 * gap.mean()),
            "bootstrap95_pp": [100.0 * x for x in ci],
            "margin_pp": 100.0 * CELL_REFERENCE_MARGIN,
            "pass": passed,
            "activation_nmse_vs_teacher_mean": float(cell_nmse[c].mean()),
            "calibration_nmse_vs_teacher_mean": float(cell_cal_nmse[c].mean()),
        }
    out["cell_reference_validity"] = validity
    out["cell_validity_pattern"] = [pass_map[c] for c in CELLS]

    b = cell_acc["N192_W32"]
    cal = cell_acc["N384_W32"]
    cap = cell_acc["N192_W64"]
    both = cell_acc["N384_W64"]
    contrasts = {
        "calibration_only_gain_N384_W32_minus_N192_W32": cal - b,
        "capacity_only_gain_N192_W64_minus_N192_W32": cap - b,
        "combined_gain_N384_W64_minus_N192_W32": both - b,
        "factorial_interaction": (both - cap) - (cal - b),
    }
    out["descriptive_factorial_contrasts"] = {}
    for name, arr in contrasts.items():
        ci = percentile95(arr[idx].mean(axis=1))
        out["descriptive_factorial_contrasts"][name] = {
            "mean_pp": float(100.0 * arr.mean()),
            "bootstrap95_pp": [100.0 * x for x in ci],
        }

    if not target_pass:
        decision = "STOP_ROBUST_TARGET_INVALID"
    elif pass_map["N192_W32"]:
        decision = "STOP_C71_BASELINE_FAILURE_NOT_REPRODUCED"
    else:
        cal_ok = pass_map["N384_W32"]
        cap_ok = pass_map["N192_W64"]
        both_ok = pass_map["N384_W64"]
        if not both_ok and (cal_ok or cap_ok):
            decision = "STOP_NONMONOTONIC_FACTORIAL_REPAIR"
        elif not both_ok:
            decision = "NO_REPAIR_AT_N384_W64"
        elif cal_ok and not cap_ok:
            decision = "LOCALIZE_CALIBRATION_QUANTITY_REPAIR"
        elif cap_ok and not cal_ok:
            decision = "LOCALIZE_MAPPING_CAPACITY_REPAIR"
        elif cal_ok and cap_ok:
            decision = "BOTH_SINGLE_FACTOR_REPAIRS"
        else:
            decision = "LOCALIZE_CALIBRATION_CAPACITY_INTERACTION_REPAIR"
    out["decision"] = decision

    out["interpretation_boundary"] = [
        "C72E is exploratory; a factor-level repair is a sufficient intervention under the tested protocol, not proof of a unique fundamental cause.",
        "The N192/W32 baseline recreates the C71 D64 direct-mapping design on a fresh cohort; if it is valid, factorial localization is stopped rather than forcing a cause.",
        "The N384 set is nested around the exact original 192 calibration subset, preventing subset replacement from masquerading as a sample-quantity effect.",
        "All cells use 600 updates, so calibration-size effects concern sample diversity/generalization under fixed update count.",
        "No reduced interface dimension is selected, and results do not transfer to imported Residual CNN C59/C60 or SmallViT."
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
