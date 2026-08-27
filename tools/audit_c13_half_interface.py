from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEEDS = list(range(1440, 1448))
BOOTSTRAP_SEED = 20260901
N_BOOT = 100_000


def pct95(x):
    q = np.quantile(x, [0.025, 0.975])
    return [float(q[0]), float(q[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("results/confirmatory/c13_half_interface/seed_rows.csv"))
    args = ap.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["seed"]))
    assert [int(r["seed"]) for r in rows] == SEEDS

    frozen = np.array([float(r["frozen_final_nmse"]) for r in rows])
    sketch = np.array([float(r["sketch32_final_nmse"]) for r in rows])
    anchored = np.array([float(r["anchored32_final_nmse"]) for r in rows])
    full = np.array([float(r["full64_final_nmse"]) for r in rows])
    aval = np.array([float(r["anchored32_val_acc"]) for r in rows])
    fval = np.array([float(r["full64_val_acc"]) for r in rows])
    atest = np.array([float(r["anchored32_test_acc"]) for r in rows])
    ftest = np.array([float(r["full64_test_acc"]) for r in rows])

    d_frozen = anchored - frozen
    d_sketch = anchored - sketch
    ratio = anchored / full
    val_diff = aval - fval
    test_diff = atest - ftest

    np.testing.assert_allclose(d_frozen, [float(r["D_frozen"]) for r in rows], rtol=0, atol=1e-12)
    np.testing.assert_allclose(d_sketch, [float(r["D_sketch"]) for r in rows], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ratio, [float(r["R_half_full"]) for r in rows], rtol=0, atol=1e-12)
    np.testing.assert_allclose(val_diff, [float(r["val_acc_diff"]) for r in rows], rtol=0, atol=1e-12)
    np.testing.assert_allclose(test_diff, [float(r["test_acc_diff"]) for r in rows], rtol=0, atol=1e-12)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(rows), size=(N_BOOT, len(rows)))
    b1 = d_frozen[idx].mean(axis=1)
    b2 = d_sketch[idx].mean(axis=1)
    b3 = np.exp(np.log(ratio)[idx].mean(axis=1))
    bv = val_diff[idx].mean(axis=1)
    bt = test_diff[idx].mean(axis=1)

    out = {
        "P1": {"mean": float(d_frozen.mean()), "bootstrap95": pct95(b1), "pass": bool(np.quantile(b1, 0.975) < 0), "all_negative": bool(np.all(d_frozen < 0))},
        "P2": {"mean": float(d_sketch.mean()), "bootstrap95": pct95(b2), "pass": bool(np.quantile(b2, 0.975) < 0), "all_negative": bool(np.all(d_sketch < 0))},
        "P3": {"geometric_mean_ratio": float(np.exp(np.log(ratio).mean())), "bootstrap95": pct95(b3), "range": [float(ratio.min()), float(ratio.max())], "pass": bool(np.quantile(b3, 0.975) < 1.35)},
        "validation_guardrail": {"mean": float(val_diff.mean()), "bootstrap95": pct95(bv), "pass": bool(np.quantile(bv, 0.025) > -0.02)},
        "test_safeguard": {"mean": float(test_diff.mean()), "bootstrap95": pct95(bt), "pass": bool(np.quantile(bt, 0.025) > -0.02)},
    }
    out["overall"] = "CONFIRMATORY_PASS" if all(out[k]["pass"] for k in ("P1", "P2", "P3", "validation_guardrail", "test_safeguard")) else "CONFIRMATORY_FAIL"
    print(json.dumps(out, indent=2))
    if out["overall"] != "CONFIRMATORY_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
