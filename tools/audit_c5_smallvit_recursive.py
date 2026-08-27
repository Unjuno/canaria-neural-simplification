from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ATTEMPTED = list(range(1350, 1362))
EXPECTED_ELIGIBLE = [1350, 1351, 1352, 1353, 1354, 1355, 1357, 1359, 1360]
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
        default=Path("results/confirmatory/c5_smallvit_recursive/seed_rows.csv"),
    )
    args = ap.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["seed"]))

    attempted = [int(r["seed"]) for r in rows]
    assert attempted == ATTEMPTED, (attempted, ATTEMPTED)
    eligible_rows = [r for r in rows if r["eligible"] == "True"]
    eligible = [int(r["seed"]) for r in eligible_rows]
    assert eligible == EXPECTED_ELIGIBLE, (eligible, EXPECTED_ELIGIBLE)
    assert len(eligible) >= MIN_ELIGIBLE

    frozen = np.array([float(r["frozen_val_nmse"]) for r in eligible_rows])
    joint = np.array([float(r["joint_val_nmse"]) for r in eligible_rows])
    direct = np.array([float(r["direct_val_nmse"]) for r in eligible_rows])
    joint_val_acc = np.array([float(r["joint_val_acc"]) for r in eligible_rows])
    direct_val_acc = np.array([float(r["direct_val_acc"]) for r in eligible_rows])
    joint_test_acc = np.array([float(r["joint_test_acc"]) for r in eligible_rows])
    direct_test_acc = np.array([float(r["direct_test_acc"]) for r in eligible_rows])

    d = joint - frozen
    ratio = joint / direct
    val_diff = joint_val_acc - direct_val_acc
    test_diff = joint_test_acc - direct_test_acc

    stored_d = np.array([float(r["D_joint_minus_frozen"]) for r in eligible_rows])
    stored_r = np.array([float(r["R_joint_over_direct"]) for r in eligible_rows])
    stored_v = np.array([float(r["val_acc_diff"]) for r in eligible_rows])
    stored_t = np.array([float(r["test_acc_diff"]) for r in eligible_rows])
    np.testing.assert_allclose(d, stored_d, rtol=0, atol=1e-12)
    np.testing.assert_allclose(ratio, stored_r, rtol=0, atol=1e-12)
    np.testing.assert_allclose(val_diff, stored_v, rtol=0, atol=1e-12)
    np.testing.assert_allclose(test_diff, stored_t, rtol=0, atol=1e-12)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(eligible_rows), size=(N_BOOT, len(eligible_rows)))
    boot_d = d[idx].mean(axis=1)
    boot_ratio = np.exp(np.log(ratio)[idx].mean(axis=1))
    boot_val = val_diff[idx].mean(axis=1)
    boot_test = test_diff[idx].mean(axis=1)

    out = {
        "eligible_count": len(eligible_rows),
        "P1": {
            "mean": float(d.mean()),
            "bootstrap95": pct95(boot_d),
            "all_negative": bool(np.all(d < 0)),
            "pass": bool(np.quantile(boot_d, 0.975) < 0),
        },
        "P2": {
            "geometric_mean_ratio": float(np.exp(np.log(ratio).mean())),
            "bootstrap95": pct95(boot_ratio),
            "range": [float(ratio.min()), float(ratio.max())],
            "pass": bool(np.quantile(boot_ratio, 0.975) < 1.20),
        },
        "P3": {
            "mean": float(val_diff.mean()),
            "bootstrap95": pct95(boot_val),
            "pass": bool(np.quantile(boot_val, 0.025) > -0.03),
        },
        "test_safeguard": {
            "mean": float(test_diff.mean()),
            "bootstrap95": pct95(boot_test),
            "pass": bool(np.quantile(boot_test, 0.025) > -0.03),
        },
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
