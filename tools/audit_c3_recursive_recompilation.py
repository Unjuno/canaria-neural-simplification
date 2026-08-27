from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = list(range(1330, 1338))
BOOTSTRAP_SEED = 20260827
N_BOOT = 100_000


def load_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r["seed"]))
    return rows


def pct95(x):
    q = np.quantile(x, [0.025, 0.975])
    return [float(q[0]), float(q[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("results/confirmatory/c3_recursive_recompilation/seed_rows.csv"),
    )
    args = ap.parse_args()
    rows = load_rows(args.csv)
    seeds = [int(r["seed"]) for r in rows]
    assert seeds == EXPECTED_SEEDS, (seeds, EXPECTED_SEEDS)

    frozen = np.array([float(r["frozen_recursive_val_nmse"]) for r in rows])
    edges = np.array([float(r["edges_recursive_val_nmse"]) for r in rows])
    joint = np.array([float(r["all_unfrozen_recursive_val_nmse"]) for r in rows])
    direct = np.array([float(r["direct_val_nmse"]) for r in rows])
    joint_test = np.array([float(r["all_unfrozen_test_acc"]) for r in rows])
    direct_test = np.array([float(r["direct_test_acc"]) for r in rows])

    d_frozen = joint - frozen
    ratios = joint / direct
    d_edges = joint - edges
    test_diff = joint_test - direct_test

    stored = {
        "D_frozen": np.array([float(r["D_frozen"]) for r in rows]),
        "R": np.array([float(r["R_recursive_over_direct"]) for r in rows]),
        "D_edges": np.array([float(r["D_edges"]) for r in rows]),
        "test_diff": np.array(
            [float(r["test_acc_diff_all_unfrozen_minus_direct"]) for r in rows]
        ),
    }
    np.testing.assert_allclose(d_frozen, stored["D_frozen"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ratios, stored["R"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(d_edges, stored["D_edges"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(test_diff, stored["test_diff"], rtol=0, atol=1e-12)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(rows), size=(N_BOOT, len(rows)))
    boot_frozen = d_frozen[idx].mean(axis=1)
    boot_ratio = np.exp(np.log(ratios)[idx].mean(axis=1))
    boot_edges = d_edges[idx].mean(axis=1)
    boot_test = test_diff[idx].mean(axis=1)

    result = {
        "P1": {
            "mean": float(d_frozen.mean()),
            "bootstrap95": pct95(boot_frozen),
            "pass": bool(np.quantile(boot_frozen, 0.975) < 0),
        },
        "P2": {
            "geometric_mean_ratio": float(np.exp(np.log(ratios).mean())),
            "bootstrap95": pct95(boot_ratio),
            "pass": bool(np.quantile(boot_ratio, 0.975) < 1.50),
        },
        "P3": {
            "mean": float(d_edges.mean()),
            "bootstrap95": pct95(boot_edges),
            "pass": bool(np.quantile(boot_edges, 0.975) < 0),
        },
        "test_safeguard": {
            "mean": float(test_diff.mean()),
            "bootstrap95": pct95(boot_test),
            "pass": bool(np.quantile(boot_test, 0.025) > -0.02),
        },
    }
    result["overall"] = "CONFIRMATORY_PASS" if all(
        result[k]["pass"] for k in ("P1", "P2", "P3", "test_safeguard")
    ) else "CONFIRMATORY_FAIL"
    print(json.dumps(result, indent=2))
    if result["overall"] != "CONFIRMATORY_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
