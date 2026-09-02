#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = tuple(range(63400, 63416))
MINIMUM_ELIGIBLE = 8
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_SEED = 2237321090
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
        "experiment": "C65R_P0_VS_P2_CONFIRMATION",
        "evidence_class": "PROSPECTIVE_CONFIRMATORY",
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
    }

    if len(eligible) < MINIMUM_ELIGIBLE:
        out["decision"] = "STOP_INSUFFICIENT_ELIGIBLE"
        return out

    p0_acc = np.asarray([float(r["p0_validation_accuracy"]) for r in eligible], dtype=float)
    p2_acc = np.asarray([float(r["p2_validation_accuracy"]) for r in eligible], dtype=float)
    p0_nmse = np.asarray([float(r["p0_nmse"]) for r in eligible], dtype=float)
    p2_nmse = np.asarray([float(r["p2_nmse"]) for r in eligible], dtype=float)

    diff = p0_acc - p2_acc
    log_ratio = np.log(p0_nmse / p2_nmse)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(eligible), size=(BOOTSTRAP_RESAMPLES, len(eligible)), endpoint=False)
    diff_boot = diff[idx].mean(axis=1)
    ratio_boot = np.exp(log_ratio[idx].mean(axis=1))
    diff_ci = percentile95(diff_boot)
    ratio_ci = percentile95(ratio_boot)
    validation_pass = bool(diff_ci[0] > VALIDATION_MARGIN)
    nmse_pass = bool(ratio_ci[1] < NMSE_RATIO_MARGIN)

    out["validation_noninferiority"] = {
        "mean_difference_fraction": float(diff.mean()),
        "mean_difference_pp": float(100.0 * diff.mean()),
        "bootstrap95_fraction": diff_ci,
        "bootstrap95_pp": [100.0 * x for x in diff_ci],
        "margin_fraction": VALIDATION_MARGIN,
        "margin_pp": 100.0 * VALIDATION_MARGIN,
        "pass": validation_pass,
    }
    out["nmse_ratio"] = {
        "geometric_mean_p0_over_p2": float(math.exp(float(log_ratio.mean()))),
        "bootstrap95": ratio_ci,
        "margin": NMSE_RATIO_MARGIN,
        "pass": nmse_pass,
    }
    out["decision"] = (
        "C65R_CONFIRMATORY_PASS" if validation_pass and nmse_pass
        else "C65R_CONFIRMATORY_FAIL"
    )
    out["interpretation_boundary"] = [
        "A PASS is non-inferiority under the exact locked C65R protocol, not equality or superiority of P0.",
        "P0 means no teacher-residual correction and no top-boundary adaptation before compilation; it does not imply zero teacher residual or representation equivalence.",
        "A PASS does not establish that teacher correction is universally unnecessary or that zero is a universal minimum interface.",
        "This C65R evidence belongs to the Residual-MLP testbed and does not confirm the imported Residual CNN C59/C60 line."
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
