#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = tuple(range(66400, 66416))
MINIMUM_ELIGIBLE = 8
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_SEED = 2190259452
CLEAN_NI_MARGIN = -0.04
SHIFT_SUPERIORITY_MARGIN = 0.0
VALIDITY_MARGIN = -0.20


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
        "experiment": "C68E_TEACHER_VALIDITY_REPAIR_EXPLORATION",
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
        out["failed_gates"] = []
        return out

    clean_clean = np.asarray([float(r["clean_teacher_clean_validation_accuracy"]) for r in eligible])
    clean_shift = np.asarray([float(r["clean_teacher_shifted_validation_accuracy"]) for r in eligible])
    aug_clean = np.asarray([float(r["augmented_teacher_clean_validation_accuracy"]) for r in eligible])
    aug_shift = np.asarray([float(r["augmented_teacher_shifted_validation_accuracy"]) for r in eligible])

    clean_diff = aug_clean - clean_clean
    shift_gain = aug_shift - clean_shift
    aug_drop = aug_shift - aug_clean
    baseline_drop = clean_shift - clean_clean

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(eligible), size=(BOOTSTRAP_RESAMPLES, len(eligible)), endpoint=False)
    clean_ci = percentile95(clean_diff[idx].mean(axis=1))
    shift_ci = percentile95(shift_gain[idx].mean(axis=1))
    validity_ci = percentile95(aug_drop[idx].mean(axis=1))
    baseline_ci = percentile95(baseline_drop[idx].mean(axis=1))

    clean_pass = bool(clean_ci[0] > CLEAN_NI_MARGIN)
    shift_pass = bool(shift_ci[0] > SHIFT_SUPERIORITY_MARGIN)
    validity_pass = bool(validity_ci[0] > VALIDITY_MARGIN)
    failed = []
    if not clean_pass:
        failed.append("clean_accuracy_noninferiority")
    if not shift_pass:
        failed.append("shifted_accuracy_superiority")
    if not validity_pass:
        failed.append("augmented_teacher_task_validity")

    out.update({
        "clean_accuracy_noninferiority": {
            "mean_difference_aug_minus_clean_fraction": float(clean_diff.mean()),
            "mean_difference_aug_minus_clean_pp": float(100.0 * clean_diff.mean()),
            "bootstrap95_fraction": clean_ci,
            "bootstrap95_pp": [100.0 * x for x in clean_ci],
            "margin_fraction": CLEAN_NI_MARGIN,
            "margin_pp": 100.0 * CLEAN_NI_MARGIN,
            "pass": clean_pass,
        },
        "shifted_accuracy_superiority": {
            "mean_difference_aug_minus_clean_fraction": float(shift_gain.mean()),
            "mean_difference_aug_minus_clean_pp": float(100.0 * shift_gain.mean()),
            "bootstrap95_fraction": shift_ci,
            "bootstrap95_pp": [100.0 * x for x in shift_ci],
            "margin_fraction": SHIFT_SUPERIORITY_MARGIN,
            "margin_pp": 0.0,
            "pass": shift_pass,
        },
        "augmented_teacher_task_validity": {
            "augmented_clean_accuracy_mean": float(aug_clean.mean()),
            "augmented_shifted_accuracy_mean": float(aug_shift.mean()),
            "mean_drop_shift_minus_clean_pp": float(100.0 * aug_drop.mean()),
            "drop_bootstrap95_pp": [100.0 * x for x in validity_ci],
            "margin_pp": 100.0 * VALIDITY_MARGIN,
            "pass": validity_pass,
        },
        "baseline_clean_teacher_descriptive": {
            "clean_accuracy_mean": float(clean_clean.mean()),
            "shifted_accuracy_mean": float(clean_shift.mean()),
            "mean_drop_shift_minus_clean_pp": float(100.0 * baseline_drop.mean()),
            "drop_bootstrap95_pp": [100.0 * x for x in baseline_ci],
        },
        "failed_gates": failed,
        "decision": (
            "ADVANCE_REPAIRED_TEACHER_TO_C69E"
            if not failed else "STOP_TEACHER_VALIDITY_REPAIR_GATES_FAILED"
        ),
        "interpretation_boundary": [
            "C68E tests one locked teacher-repair recipe and is exploratory.",
            "An advance decision means the repaired teacher is a candidate valid target for a separate interface experiment; it does not imply any P0/P2 result.",
            "The clean -4 pp and task-validity -20 pp margins are preregistered experiment safeguards, not universal constants.",
            "Held-out test data are unused.",
            "This is repository Residual-MLP evidence and does not confirm the imported Residual CNN C59/C60 line."
        ],
    })
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
