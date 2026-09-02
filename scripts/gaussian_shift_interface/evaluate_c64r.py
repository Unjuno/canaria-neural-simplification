#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = tuple(range(62400, 62416))
MINIMUM_ELIGIBLE = 8
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_SEED = 567883594
VALIDATION_MARGIN = -0.02
NMSE_RATIO_MARGIN = 1.25


def percentile95(x: np.ndarray) -> list[float]:
    q = np.percentile(x, [2.5, 97.5])
    return [float(q[0]), float(q[1])]


def cond_array(rows: list[dict], key: str, metric: str) -> np.ndarray:
    return np.asarray([float(r["conditions"][key][metric]) for r in rows], dtype=float)


def paired_summary(rows: list[dict], first: str, second: str, idx: np.ndarray) -> dict:
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


def add_gates(summary: dict) -> dict:
    summary = dict(summary)
    summary["validation_margin_fraction"] = VALIDATION_MARGIN
    summary["validation_margin_pp"] = 100.0 * VALIDATION_MARGIN
    summary["validation_noninferiority_pass"] = bool(
        summary["validation_bootstrap95_fraction"][0] > VALIDATION_MARGIN
    )
    summary["nmse_ratio_margin"] = NMSE_RATIO_MARGIN
    summary["nmse_ratio_pass"] = bool(
        summary["nmse_ratio_bootstrap95"][1] < NMSE_RATIO_MARGIN
    )
    summary["joint_pass"] = bool(
        summary["validation_noninferiority_pass"] and summary["nmse_ratio_pass"]
    )
    return summary


def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None
    rx = rankdata_average(np.asarray(x, dtype=float))
    ry = rankdata_average(np.asarray(y, dtype=float))
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def geom_array(rows: list[dict], split: str, pk: str, metric: str) -> np.ndarray:
    return np.asarray([
        float(r["diagnostics"]["task_weighted_geometry"][split][pk][metric])
        for r in rows
    ], dtype=float)


def summary_stats(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def evaluate(payload: dict) -> dict:
    rows = list(payload.get("rows", []))
    by_seed = {int(r["seed"]): r for r in rows if "seed" in r}
    missing = [
        s for s in EXPECTED_SEEDS
        if s not in by_seed or by_seed[s].get("failure") == "missing_seed_artifact"
    ]
    eligible = [
        by_seed[s] for s in EXPECTED_SEEDS
        if s in by_seed and bool(by_seed[s].get("eligible", False))
    ]

    out = {
        "experiment": "C64R_P0_P1_P2_TASK_WEIGHTED_FRONTIER_EXPLORATION",
        "evidence_class": "PROSPECTIVE_EXPLORATORY",
        "expected_seeds": list(EXPECTED_SEEDS),
        "ledger_rows": len(rows),
        "artifact_rows_present": sum(
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
        return out

    n = len(eligible)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n), endpoint=False)

    p0 = add_gates(paired_summary(eligible, "p0", "p2", idx))
    p1 = add_gates(paired_summary(eligible, "p1", "p2", idx))
    p0_vs_p1 = paired_summary(eligible, "p0", "p1", idx)
    out["primary_exploratory_frontier"] = {
        "p0_vs_p2": p0,
        "p1_vs_p2": p1,
        "p0_vs_p1_descriptive": p0_vs_p1,
    }

    if p0["joint_pass"]:
        out["decision"] = "ADVANCE_P0_TO_C65R"
    elif p1["joint_pass"]:
        out["decision"] = "ADVANCE_P1_TO_C65R"
    else:
        out["decision"] = "STOP_LOWER_FRONTIER_AT_P2"

    metric_names = (
        "euclidean_capture_fraction",
        "logit_l2_retained_ratio",
        "fisher_retained_ratio",
    )
    features: dict[str, np.ndarray] = {}
    for split in ("shifted_calibration", "shifted_validation"):
        for pk in ("p0", "p1", "p2"):
            for metric in metric_names:
                features[f"{split}_{pk}_{metric}"] = geom_array(
                    eligible, split, pk, metric
                )

    out["task_weighted_geometry_summary"] = {
        name: summary_stats(vals) for name, vals in features.items()
    }

    p1_acc_gap = cond_array(eligible, "p1", "validation_accuracy") - cond_array(
        eligible, "p2", "validation_accuracy"
    )
    p1_log_nmse = np.log(
        cond_array(eligible, "p1", "nmse") / cond_array(eligible, "p2", "nmse")
    )
    p0_acc_gap = cond_array(eligible, "p0", "validation_accuracy") - cond_array(
        eligible, "p2", "validation_accuracy"
    )
    p0_log_nmse = np.log(
        cond_array(eligible, "p0", "nmse") / cond_array(eligible, "p2", "nmse")
    )

    correlation_features = {
        "validation_p1_euclidean_capture": features["shifted_validation_p1_euclidean_capture_fraction"],
        "validation_p1_logit_l2_retained": features["shifted_validation_p1_logit_l2_retained_ratio"],
        "validation_p1_fisher_retained": features["shifted_validation_p1_fisher_retained_ratio"],
        "validation_p2_euclidean_capture": features["shifted_validation_p2_euclidean_capture_fraction"],
        "validation_p2_logit_l2_retained": features["shifted_validation_p2_logit_l2_retained_ratio"],
        "validation_p2_fisher_retained": features["shifted_validation_p2_fisher_retained_ratio"],
        "validation_fisher_retained_gap_p1_minus_p2": (
            features["shifted_validation_p1_fisher_retained_ratio"]
            - features["shifted_validation_p2_fisher_retained_ratio"]
        ),
        "validation_logit_l2_retained_gap_p1_minus_p2": (
            features["shifted_validation_p1_logit_l2_retained_ratio"]
            - features["shifted_validation_p2_logit_l2_retained_ratio"]
        ),
    }
    out["mechanism_spearman_correlations"] = {
        name: {
            "with_p1_minus_p2_validation": spearman(vals, p1_acc_gap),
            "with_log_p1_over_p2_nmse": spearman(vals, p1_log_nmse),
            "with_p0_minus_p2_validation": spearman(vals, p0_acc_gap),
            "with_log_p0_over_p2_nmse": spearman(vals, p0_log_nmse),
        }
        for name, vals in correlation_features.items()
    }

    out["interpretation_boundary"] = [
        "C64R is exploratory; an advance decision selects a candidate for a separate fresh confirmatory experiment.",
        "P0 is the frozen base hierarchy compiled directly, not a zero-target AdamW adaptation condition.",
        "Task-weighted geometry analyses are mechanism-generating and are not multiplicity-corrected confirmatory tests.",
        "This experiment is Residual-MLP evidence and does not confirm the imported Residual CNN C59/C60 line."
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
