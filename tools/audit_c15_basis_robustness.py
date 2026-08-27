from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEEDS = list(range(1460, 1468))
N_BOOT = 100_000
BOOTSTRAP_SEED = 20260923
BASES = ["identity_first32", "random_20260920", "random_20260921", "random_20260922"]


def pct95(x):
    q = np.quantile(x, [0.025, 0.975])
    return [float(q[0]), float(q[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("results/confirmatory/c15_basis_robustness/seed_rows.csv"))
    args = ap.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["seed"]))
    assert [int(r["seed"]) for r in rows] == SEEDS

    for r in rows:
        nm = np.array([float(r[f"{b}_nmse"]) for b in BASES])
        np.testing.assert_allclose(float(r["D_worst"]), nm.max() - float(r["frozen_nmse"]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(float(r["R_worst"]), nm.max() / float(r["full64_nmse"]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(float(r["basis_spread"]), nm.max() / nm.min(), rtol=0, atol=1e-12)

    D = np.array([float(r["D_worst"]) for r in rows])
    R = np.array([float(r["R_worst"]) for r in rows])
    S = np.array([float(r["basis_spread"]) for r in rows])
    V = np.array([float(r["val_acc_diff_worst_minus_full"]) for r in rows])
    T = np.array([float(r["test_acc_diff_worst_minus_full"]) for r in rows])

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(rows), size=(N_BOOT, len(rows)))
    bD = D[idx].mean(axis=1)
    bR = np.exp(np.log(R)[idx].mean(axis=1))
    bS = np.exp(np.log(S)[idx].mean(axis=1))
    bV = V[idx].mean(axis=1)
    bT = T[idx].mean(axis=1)

    out = {
        "P1": {"mean": float(D.mean()), "bootstrap95": pct95(bD), "all_negative": bool(np.all(D < 0)), "pass": bool(np.quantile(bD, 0.975) < 0)},
        "P2": {"geometric_mean_ratio": float(np.exp(np.log(R).mean())), "bootstrap95": pct95(bR), "range": [float(R.min()), float(R.max())], "pass": bool(np.quantile(bR, 0.975) < 1.30)},
        "P3": {"geometric_mean_spread": float(np.exp(np.log(S).mean())), "bootstrap95": pct95(bS), "range": [float(S.min()), float(S.max())], "pass": bool(np.quantile(bS, 0.975) < 1.15)},
        "validation": {"mean": float(V.mean()), "bootstrap95": pct95(bV), "pass": bool(np.quantile(bV, 0.025) > -0.02)},
        "test": {"mean": float(T.mean()), "bootstrap95": pct95(bT), "pass": bool(np.quantile(bT, 0.025) > -0.02)},
    }
    out["overall"] = "CONFIRMATORY_PASS" if all(out[k]["pass"] for k in ("P1", "P2", "P3", "validation", "test")) else "CONFIRMATORY_FAIL"
    print(json.dumps(out, indent=2))
    if out["overall"] != "CONFIRMATORY_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
