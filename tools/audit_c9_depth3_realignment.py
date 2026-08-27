from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = list(range(1400, 1408))
BOOTSTRAP_SEED = 20260830
N_BOOT = 100_000


def pct95(x):
    q = np.quantile(x, [0.025, 0.975])
    return [float(q[0]), float(q[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("results/confirmatory/c9_depth3_realignment/seed_rows.csv"),
    )
    args = ap.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in rows]
    assert seeds == EXPECTED_SEEDS, (seeds, EXPECTED_SEEDS)

    strict = np.array([float(r["strict_final_nmse"]) for r in rows])
    realign = np.array([float(r["realign_final_nmse"]) for r in rows])
    single = np.array([float(r["single_final_nmse"]) for r in rows])
    direct = np.array([float(r["direct_final_nmse"]) for r in rows])
    strict_l2 = np.array([float(r["strict_level2_full_nmse"]) for r in rows])
    realign_l2 = np.array([float(r["realign_level2_full_nmse"]) for r in rows])
    realign_val = np.array([float(r["realign_val_acc"]) for r in rows])
    direct_val = np.array([float(r["direct_val_acc"]) for r in rows])
    realign_test = np.array([float(r["realign_test_acc"]) for r in rows])
    direct_test = np.array([float(r["direct_test_acc"]) for r in rows])

    d_strict = realign - strict
    ratio = realign / direct
    d_single = realign - single
    d_level2 = realign_l2 - strict_l2
    val_diff = realign_val - direct_val
    test_diff = realign_test - direct_test

    stored = {
        "D_strict": np.array([float(r["D_strict"]) for r in rows]),
        "R_realign": np.array([float(r["R_realign"]) for r in rows]),
        "D_single": np.array([float(r["D_single"]) for r in rows]),
        "D_level2": np.array([float(r["D_level2"]) for r in rows]),
        "val_acc_diff": np.array([float(r["val_acc_diff"]) for r in rows]),
        "test_acc_diff": np.array([float(r["test_acc_diff"]) for r in rows]),
    }
    np.testing.assert_allclose(d_strict, stored["D_strict"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ratio, stored["R_realign"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(d_single, stored["D_single"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(d_level2, stored["D_level2"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(val_diff, stored["val_acc_diff"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(test_diff, stored["test_acc_diff"], rtol=0, atol=1e-12)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(rows), size=(N_BOOT, len(rows)))
    boot1 = d_strict[idx].mean(axis=1)
    boot2 = np.exp(np.log(ratio)[idx].mean(axis=1))
    boot3 = d_single[idx].mean(axis=1)
    boot4 = d_level2[idx].mean(axis=1)
    bootv = val_diff[idx].mean(axis=1)
    boott = test_diff[idx].mean(axis=1)

    out = {
        "P1": {
            "mean": float(d_strict.mean()),
            "bootstrap95": pct95(boot1),
            "all_negative": bool(np.all(d_strict < 0)),
            "pass": bool(np.quantile(boot1, 0.975) < 0),
        },
        "P2": {
            "geometric_mean_ratio": float(np.exp(np.log(ratio).mean())),
            "bootstrap95": pct95(boot2),
            "pass": bool(np.quantile(boot2, 0.975) < 1.40),
        },
        "P3": {
            "mean": float(d_single.mean()),
            "bootstrap95": pct95(boot3),
            "all_negative": bool(np.all(d_single < 0)),
            "pass": bool(np.quantile(boot3, 0.975) < 0),
        },
        "P4": {
            "mean": float(d_level2.mean()),
            "bootstrap95": pct95(boot4),
            "all_negative": bool(np.all(d_level2 < 0)),
            "pass": bool(np.quantile(boot4, 0.975) < 0),
        },
        "validation_guardrail": {
            "mean": float(val_diff.mean()),
            "bootstrap95": pct95(bootv),
            "pass": bool(np.quantile(bootv, 0.025) > -0.02),
        },
        "test_safeguard": {
            "mean": float(test_diff.mean()),
            "bootstrap95": pct95(boott),
            "pass": bool(np.quantile(boott, 0.025) > -0.02),
        },
    }
    out["overall"] = "CONFIRMATORY_PASS" if all(
        out[k]["pass"]
        for k in ("P1", "P2", "P3", "P4", "validation_guardrail", "test_safeguard")
    ) else "CONFIRMATORY_FAIL"
    print(json.dumps(out, indent=2))
    if out["overall"] != "CONFIRMATORY_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
