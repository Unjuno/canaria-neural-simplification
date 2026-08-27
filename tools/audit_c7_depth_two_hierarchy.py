from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = list(range(1380, 1390))
MIN_ELIGIBLE = 8
BOOTSTRAP_SEED = 20260828
N_BOOT = 100_000


def pct95(x):
    q = np.quantile(x, [0.025, 0.975])
    return [float(q[0]), float(q[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("results/confirmatory/c7_depth_two_hierarchy/seed_rows.csv"),
    )
    args = ap.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["seed"]))
    assert [int(r["seed"]) for r in rows] == EXPECTED_SEEDS
    eligible = [r for r in rows if r["eligible"] == "True"]
    assert len(eligible) >= MIN_ELIGIBLE

    no_adapt = np.array([float(r["no_adapt_val_nmse"]) for r in eligible])
    joint = np.array([float(r["joint_val_nmse"]) for r in eligible])
    flat = np.array([float(r["flat_val_nmse"]) for r in eligible])
    joint_val = np.array([float(r["joint_val_acc"]) for r in eligible])
    flat_val = np.array([float(r["flat_val_acc"]) for r in eligible])
    joint_test = np.array([float(r["joint_test_acc"]) for r in eligible])
    flat_test = np.array([float(r["flat_test_acc"]) for r in eligible])

    d = joint - no_adapt
    ratio = joint / flat
    val_diff = joint_val - flat_val
    test_diff = joint_test - flat_test

    np.testing.assert_allclose(d, np.array([float(r["D_joint_minus_no_adapt"]) for r in eligible]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(ratio, np.array([float(r["R_joint_over_flat"]) for r in eligible]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(val_diff, np.array([float(r["val_acc_diff"]) for r in eligible]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(test_diff, np.array([float(r["test_acc_diff"]) for r in eligible]), rtol=0, atol=1e-12)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(eligible), size=(N_BOOT, len(eligible)))
    boot_d = d[idx].mean(axis=1)
    boot_ratio = np.exp(np.log(ratio)[idx].mean(axis=1))
    boot_val = val_diff[idx].mean(axis=1)
    boot_test = test_diff[idx].mean(axis=1)

    out = {
        "eligible_count": len(eligible),
        "P1": {"mean": float(d.mean()), "bootstrap95": pct95(boot_d), "all_negative": bool(np.all(d < 0)), "pass": bool(np.quantile(boot_d, 0.975) < 0)},
        "P2": {"geometric_mean_ratio": float(np.exp(np.log(ratio).mean())), "bootstrap95": pct95(boot_ratio), "range": [float(ratio.min()), float(ratio.max())], "pass": bool(np.quantile(boot_ratio, 0.975) < 1.10)},
        "P3": {"mean": float(val_diff.mean()), "bootstrap95": pct95(boot_val), "pass": bool(np.quantile(boot_val, 0.025) > -0.02)},
        "test_safeguard": {"mean": float(test_diff.mean()), "bootstrap95": pct95(boot_test), "pass": bool(np.quantile(boot_test, 0.025) > -0.02)},
    }
    out["overall"] = "CONFIRMATORY_PASS" if (
        out["eligible_count"] >= MIN_ELIGIBLE
        and all(out[k]["pass"] for k in ("P1", "P2", "P3", "test_safeguard"))
    ) else "CONFIRMATORY_FAIL"
    print(json.dumps(out, indent=2))
    if out["overall"] != "CONFIRMATORY_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
