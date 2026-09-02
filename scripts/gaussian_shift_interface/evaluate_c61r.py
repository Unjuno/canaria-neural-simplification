#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np

EXPECTED_SEEDS = tuple(range(59400, 59416))
MIN_ELIGIBLE = 8
VAL_MARGIN = -0.02
NMSE_RATIO_MARGIN = 1.25
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 3960519679


def percentile_ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    chunk: int = 10_000,
) -> np.ndarray:
    n = len(x)
    if n == 0:
        raise ValueError("cannot bootstrap zero eligible rows")
    out = np.empty(n_resamples, dtype=np.float64)
    cursor = 0
    while cursor < n_resamples:
        m = min(chunk, n_resamples - cursor)
        idx = rng.integers(0, n, size=(m, n), endpoint=False)
        batch = x[idx]
        if statistic is np.mean:
            vals = batch.mean(axis=1)
        else:
            vals = np.asarray([statistic(row) for row in batch], dtype=np.float64)
        out[cursor:cursor + m] = vals
        cursor += m
    return out


def geometric_mean(x: np.ndarray) -> float:
    if np.any(x <= 0):
        raise ValueError("geometric mean requires strictly positive values")
    return float(math.exp(np.log(x).mean()))


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError("input must be a list or an object containing rows")
    return rows


def validate_rows(rows: list[dict]) -> None:
    seeds = [int(r["seed"]) for r in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed rows")
    unexpected = sorted(set(seeds) - set(EXPECTED_SEEDS))
    if unexpected:
        raise ValueError(f"unexpected C61R seeds: {unexpected}")
    for r in rows:
        if not bool(r.get("eligible", False)):
            continue
        for key in (
            "p4_validation_accuracy",
            "p8_validation_accuracy",
            "p4_nmse",
            "p8_nmse",
        ):
            if key not in r:
                raise ValueError(f"eligible seed {r['seed']} missing {key}")
        if not (0.0 <= float(r["p4_validation_accuracy"]) <= 1.0):
            raise ValueError(f"invalid P4 accuracy at seed {r['seed']}")
        if not (0.0 <= float(r["p8_validation_accuracy"]) <= 1.0):
            raise ValueError(f"invalid P8 accuracy at seed {r['seed']}")
        if float(r["p4_nmse"]) <= 0.0 or float(r["p8_nmse"]) <= 0.0:
            raise ValueError(f"NMSE must be positive at seed {r['seed']}")


def evaluate(rows: list[dict]) -> dict:
    validate_rows(rows)
    eligible = [r for r in rows if bool(r.get("eligible", False))]
    result = {
        "experiment": "C61R_GAUSSIAN_SHIFT_P4_VS_P8_PROSPECTIVE_REPLICATION",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "attempted_rows_present": len(rows),
        "eligible_count": len(eligible),
        "minimum_eligible": MIN_ELIGIBLE,
        "expected_seeds": list(EXPECTED_SEEDS),
        "missing_seed_rows": sorted(set(EXPECTED_SEEDS) - {int(r['seed']) for r in rows}),
    }
    if len(eligible) < MIN_ELIGIBLE:
        result["decision"] = "STOP_INSUFFICIENT_ELIGIBLE"
        return result

    val_diff = np.asarray([
        float(r["p4_validation_accuracy"]) - float(r["p8_validation_accuracy"])
        for r in eligible
    ], dtype=np.float64)
    nmse_ratio = np.asarray([
        float(r["p4_nmse"]) / float(r["p8_nmse"])
        for r in eligible
    ], dtype=np.float64)

    ss = np.random.SeedSequence(BOOTSTRAP_SEED)
    rng_val, rng_ratio = [np.random.default_rng(s) for s in ss.spawn(2)]
    val_ci = percentile_ci(paired_bootstrap(val_diff, np.mean, rng_val))
    ratio_ci = percentile_ci(paired_bootstrap(nmse_ratio, geometric_mean, rng_ratio))

    val_point = float(val_diff.mean())
    ratio_point = geometric_mean(nmse_ratio)
    val_pass = val_ci[0] > VAL_MARGIN
    ratio_pass = ratio_ci[1] < NMSE_RATIO_MARGIN

    result["validation_noninferiority"] = {
        "mean_difference_fraction": val_point,
        "mean_difference_pp": 100.0 * val_point,
        "bootstrap95_fraction": list(val_ci),
        "bootstrap95_pp": [100.0 * val_ci[0], 100.0 * val_ci[1]],
        "margin_fraction": VAL_MARGIN,
        "margin_pp": 100.0 * VAL_MARGIN,
        "pass": bool(val_pass),
    }
    result["nmse_ratio"] = {
        "geometric_mean_p4_over_p8": ratio_point,
        "bootstrap95": list(ratio_ci),
        "margin": NMSE_RATIO_MARGIN,
        "pass": bool(ratio_pass),
    }
    if all("frozen_nmse" in r for r in eligible):
        delta = np.asarray([
            float(r["p4_nmse"]) - float(r["frozen_nmse"])
            for r in eligible
        ], dtype=np.float64)
        result["informative_p4_delta_nmse_vs_frozen_mean"] = float(delta.mean())

    result["decision"] = (
        "C61R_CONFIRMATORY_PASS" if val_pass and ratio_pass else "C61R_CONFIRMATORY_FAIL"
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = evaluate(load_rows(args.input))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
