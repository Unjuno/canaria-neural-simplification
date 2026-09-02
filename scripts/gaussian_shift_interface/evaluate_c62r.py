#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = tuple(range(60400, 60416))
MINIMUM_ELIGIBLE = 8
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_SEED = 2842827726
VALIDATION_MARGIN = -0.02
NMSE_RATIO_MARGIN = 1.25


def percentile95(x: np.ndarray) -> list[float]:
    q = np.percentile(x, [2.5, 97.5])
    return [float(q[0]), float(q[1])]


def rankdata_average(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def cond_array(rows: list[dict], key: str, metric: str) -> np.ndarray:
    return np.asarray([float(r["conditions"][key][metric]) for r in rows], dtype=float)


def paired_summary(
    rows: list[dict], first: str, second: str, idx: np.ndarray
) -> dict:
    a_acc = cond_array(rows, first, "validation_accuracy")
    b_acc = cond_array(rows, second, "validation_accuracy")
    diff = a_acc - b_acc
    diff_boot = diff[idx].mean(axis=1)

    a_nmse = cond_array(rows, first, "nmse")
    b_nmse = cond_array(rows, second, "nmse")
    log_ratio = np.log(a_nmse / b_nmse)
    ratio_boot = np.exp(log_ratio[idx].mean(axis=1))
    return {
        "first": first,
        "second": second,
        "validation_difference_first_minus_second_fraction": float(diff.mean()),
        "validation_difference_first_minus_second_pp": float(100.0 * diff.mean()),
        "validation_bootstrap95_fraction": percentile95(diff_boot),
        "validation_bootstrap95_pp": [100.0 * x for x in percentile95(diff_boot)],
        "nmse_geometric_mean_ratio_first_over_second": float(math.exp(float(log_ratio.mean()))),
        "nmse_ratio_bootstrap95": percentile95(ratio_boot),
    }


def nested_feature(rows: list[dict], path: tuple[str, ...]) -> np.ndarray:
    vals = []
    for row in rows:
        cur = row["diagnostics"]
        for key in path:
            cur = cur[key]
        vals.append(float(cur))
    return np.asarray(vals, dtype=float)


def evaluate(payload: dict) -> dict:
    rows = list(payload.get("rows", []))
    by_seed = {int(r["seed"]): r for r in rows if "seed" in r}
    missing = [s for s in EXPECTED_SEEDS if s not in by_seed or by_seed[s].get("failure") == "missing_seed_artifact"]
    eligible = [by_seed[s] for s in EXPECTED_SEEDS if s in by_seed and bool(by_seed[s].get("eligible", False))]

    out = {
        "experiment": "C62R_P2_FRONTIER_AND_BASIS_MECHANISM_EXPLORATION",
        "evidence_class": "PROSPECTIVE_EXPLORATORY",
        "expected_seeds": list(EXPECTED_SEEDS),
        "ledger_rows": len(rows),
        "artifact_rows_present": sum(1 for s in EXPECTED_SEEDS if s in by_seed and by_seed[s].get("failure") != "missing_seed_artifact"),
        "eligible_count": len(eligible),
        "minimum_eligible": MINIMUM_ELIGIBLE,
        "missing_seed_rows": missing,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confirmatory_claim_allowed": False,
    }

    if len(eligible) < MINIMUM_ELIGIBLE:
        out["decision"] = "STOP_INSUFFICIENT_ELIGIBLE"
        return out

    n = len(eligible)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n), endpoint=False)

    primary = paired_summary(eligible, "qr_p2", "qr_p4", idx)
    primary["validation_margin_fraction"] = VALIDATION_MARGIN
    primary["validation_margin_pp"] = 100.0 * VALIDATION_MARGIN
    primary["validation_noninferiority_pass"] = bool(
        primary["validation_bootstrap95_fraction"][0] > VALIDATION_MARGIN
    )
    primary["nmse_ratio_margin"] = NMSE_RATIO_MARGIN
    primary["nmse_ratio_pass"] = bool(
        primary["nmse_ratio_bootstrap95"][1] < NMSE_RATIO_MARGIN
    )
    out["primary_exploratory_frontier"] = primary

    if primary["validation_noninferiority_pass"] and primary["nmse_ratio_pass"]:
        out["decision"] = "ADVANCE_QR_P2_TO_C63R"
    else:
        out["decision"] = "STOP_P2_FRONTIER_AT_C62R"

    comparisons = [
        ("qr_p2", "random_p2"),
        ("qr_p4", "random_p4"),
        ("svd_p2", "qr_p2"),
        ("svd_p4", "qr_p4"),
    ]
    out["mechanism_basis_controls"] = {
        f"{a}_vs_{b}": paired_summary(eligible, a, b, idx) for a, b in comparisons
    }

    feature_paths = {
        "entropy_effective_rank": ("spectrum", "entropy_effective_rank"),
        "stable_rank": ("spectrum", "stable_rank"),
        "optimal_svd_energy_fraction_k2": ("spectrum", "optimal_svd_energy_fraction_k2"),
        "optimal_svd_energy_fraction_k4": ("spectrum", "optimal_svd_energy_fraction_k4"),
        "optimal_svd_energy_fraction_k8": ("spectrum", "optimal_svd_energy_fraction_k8"),
        "qr_calibration_capture_k2": ("energy_capture", "qr", "k2", "calibration"),
        "qr_calibration_capture_k4": ("energy_capture", "qr", "k4", "calibration"),
        "qr_validation_capture_k2": ("energy_capture", "qr", "k2", "shifted_validation"),
        "qr_validation_capture_k4": ("energy_capture", "qr", "k4", "shifted_validation"),
        "svd_validation_capture_k2": ("energy_capture", "svd", "k2", "shifted_validation"),
        "svd_validation_capture_k4": ("energy_capture", "svd", "k4", "shifted_validation"),
        "random_validation_capture_k2": ("energy_capture", "random", "k2", "shifted_validation"),
        "random_validation_capture_k4": ("energy_capture", "random", "k4", "shifted_validation"),
        "qr_svd_alignment_k2": ("qr_to_svd_alignment", "k2", "mean_squared_principal_cosine"),
        "qr_svd_alignment_k4": ("qr_to_svd_alignment", "k4", "mean_squared_principal_cosine"),
    }
    features = {name: nested_feature(eligible, path) for name, path in feature_paths.items()}
    features["qr_validation_capture_gap_k4_minus_k2"] = (
        features["qr_validation_capture_k4"] - features["qr_validation_capture_k2"]
    )

    out["mechanism_feature_summary"] = {
        name: {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        for name, vals in features.items()
    }

    qr2_acc = cond_array(eligible, "qr_p2", "validation_accuracy")
    qr4_acc = cond_array(eligible, "qr_p4", "validation_accuracy")
    frontier_val = qr2_acc - qr4_acc
    qr2_nmse = cond_array(eligible, "qr_p2", "nmse")
    qr4_nmse = cond_array(eligible, "qr_p4", "nmse")
    frontier_log_nmse = np.log(qr2_nmse / qr4_nmse)

    out["mechanism_spearman_correlations"] = {
        name: {
            "with_qr_p2_minus_p4_validation": spearman(vals, frontier_val),
            "with_log_qr_p2_over_p4_nmse": spearman(vals, frontier_log_nmse),
        }
        for name, vals in features.items()
    }

    frozen = np.asarray([float(r["frozen_nmse"]) for r in eligible], dtype=float)
    out["informative_qr_p2_delta_nmse_vs_frozen_mean"] = float(np.mean(qr2_nmse - frozen))
    out["interpretation_boundary"] = [
        "The advance decision is exploratory and cannot support a P2 sufficiency claim.",
        "Basis-control and correlation analyses are mechanism-generating and are not multiplicity-corrected confirmatory tests.",
        "This experiment follows the C61R Residual-MLP testbed and is not evidence for the imported Residual CNN C59/C60 line."
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
