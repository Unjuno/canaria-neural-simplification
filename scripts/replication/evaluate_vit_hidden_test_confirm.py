import argparse, json, math
from pathlib import Path

import numpy as np

SEEDS = list(range(9200, 9216))
BOOTSTRAP_RESAMPLES = 100000
BOOTSTRAP_SEED = 20260912
MIN_ELIGIBLE = 12
LOWER_FRACTION_COMPOSED_SMALLER = 0.75
TEST_NI_MARGIN = -0.03


def bootstrap_mean(values, rng, n=BOOTSTRAP_RESAMPLES):
    a = np.asarray(values, dtype=np.float64)
    if len(a) == 0:
        return [None, None]
    idx = rng.integers(0, len(a), size=(n, len(a)))
    means = a[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def audit_firewall(row):
    errors = []
    tp = row.get("test_policy", {})
    if tp.get("candidate_test_metrics_computed_preselection") is not False:
        errors.append("preselection candidate test metrics flag not false")
    if tp.get("test_evaluations_before_selection") != 0:
        errors.append("test evaluations occurred before selection")
    if row.get("decision_state") == "COMPLETE" and tp.get("actual_test_evaluations_after_selection") != 3:
        errors.append("complete row did not perform exactly three post-selection test evaluations")
    for group in ("componentwise", "composed"):
        for cand in row.get(group, []):
            if cand.get("test_acc") is not None:
                errors.append(f"{group} candidate contains preselection test metric")
                break
    for name in ("selected_componentwise_pretest", "selected_composed_pretest"):
        obj = row.get(name)
        if obj is not None and obj.get("test_acc") is not None:
            errors.append(name + " contains test metric")
    return errors


def evaluate(root):
    root = Path(root)
    rows = []
    errors = []
    for seed in SEEDS:
        p = root / f"seed_{seed}.json"
        if not p.exists():
            errors.append(f"missing seed file {seed}")
            continue
        row = json.loads(p.read_text())
        if row.get("seed") != seed:
            errors.append(f"seed identity mismatch {seed}")
        fw = audit_firewall(row)
        errors += [f"seed {seed}: {e}" for e in fw]
        rows.append(row)

    eligible = [r for r in rows if r.get("eligible") is True]
    complete = [r for r in eligible if r.get("decision_state") == "COMPLETE"]
    ineligible = [r["seed"] for r in rows if r.get("eligible") is False]
    missing_endpoint = [r["seed"] for r in eligible if r.get("decision_state") != "COMPLETE"]

    log_ratios = []
    test_diffs = []
    composed_test_utils = []
    selected = []
    for r in complete:
        sep = r["selected_componentwise"]
        comp = r["selected_composed"]
        br = comp["replacement_params"] / sep["replacement_params"]
        if not (br > 0 and np.isfinite(br)):
            errors.append(f"seed {r['seed']}: invalid budget ratio {br}")
            continue
        lr = math.log2(br)
        log_ratios.append(lr)
        test_diffs.append(comp["test_acc"] - sep["test_acc"])
        composed_test_utils.append(comp["test_utility"])
        selected.append({
            "seed": r["seed"],
            "componentwise_budget": sep["replacement_params"],
            "composed_budget": comp["replacement_params"],
            "budget_ratio": br,
            "log2_budget_ratio": lr,
            "componentwise_test_acc": sep["test_acc"],
            "composed_test_acc": comp["test_acc"],
            "test_diff_composed_minus_componentwise": comp["test_acc"] - sep["test_acc"],
            "composed_test_utility": comp["test_utility"]
        })

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    budget_ci = bootstrap_mean(log_ratios, rng) if log_ratios else [None, None]
    test_diff_ci = bootstrap_mean(test_diffs, rng) if test_diffs else [None, None]
    test_utility_ci = bootstrap_mean(composed_test_utils, rng) if composed_test_utils else [None, None]
    smaller = sum(x < 0 for x in log_ratios)
    ties = sum(x == 0 for x in log_ratios)
    larger = sum(x > 0 for x in log_ratios)
    required_smaller = math.ceil(LOWER_FRACTION_COMPOSED_SMALLER * len(complete)) if complete else None

    coverage_pass = len(rows) == len(SEEDS) and len(eligible) >= MIN_ELIGIBLE and len(complete) == len(eligible)
    budget_pass = bool(coverage_pass and budget_ci[1] is not None and budget_ci[1] < 0 and smaller >= required_smaller)
    test_ni_pass = bool(coverage_pass and test_diff_ci[0] is not None and test_diff_ci[0] > TEST_NI_MARGIN)
    firewall_pass = not any("test" in e.lower() for e in errors)

    if errors:
        decision = "INVALID_AUDIT_ERROR"
    elif not coverage_pass:
        decision = "UNCERTAIN_COVERAGE_OR_ENDPOINT"
    elif budget_pass and test_ni_pass and firewall_pass:
        decision = "SMALLVIT_HIDDEN_TEST_CONFIRMATORY_PASS"
    elif not budget_pass:
        decision = "SMALLVIT_HIDDEN_TEST_CONFIRMATORY_FAIL_BUDGET"
    else:
        decision = "SMALLVIT_HIDDEN_TEST_PRIMARY_PASS_TASK_SAFEGUARD_UNCERTAIN"

    return {
        "schema_version": 1,
        "experiment": "SMALLVIT_HIDDEN_TEST_CONFIRMATORY_V1",
        "decision": decision,
        "seed_plan": SEEDS,
        "attempted": len(rows),
        "eligible": len(eligible),
        "ineligible_seeds": ineligible,
        "complete_eligible": len(complete),
        "missing_endpoint_seeds": missing_endpoint,
        "primary_budget": {
            "mean_log2_ratio": float(np.mean(log_ratios)) if log_ratios else None,
            "bootstrap95_mean_log2_ratio": budget_ci,
            "geometric_composed_over_componentwise_ratio": float(2 ** np.mean(log_ratios)) if log_ratios else None,
            "composed_lower": smaller,
            "ties": ties,
            "componentwise_lower": larger,
            "required_composed_lower": required_smaller,
            "pass": budget_pass
        },
        "postselection_test_safeguard": {
            "mean_composed_minus_componentwise_accuracy": float(np.mean(test_diffs)) if test_diffs else None,
            "bootstrap95_mean_difference": test_diff_ci,
            "noninferiority_margin": TEST_NI_MARGIN,
            "pass": test_ni_pass,
            "mean_composed_test_utility": float(np.mean(composed_test_utils)) if composed_test_utils else None,
            "bootstrap95_mean_composed_test_utility": test_utility_ci
        },
        "test_firewall_pass": firewall_pass,
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "rng_seed": BOOTSTRAP_SEED, "method": "paired percentile seed bootstrap"},
        "selected_rows": selected,
        "errors": errors,
        "interpretation_boundary": "One fixed sklearn-digits split and SmallViT family. Model seeds are not independent datasets. Test was evaluated only after selection for baseline and the two selected endpoints; no candidate-grid test metrics were computed."
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = evaluate(args.input)
    Path(args.out).write_text(json.dumps(d, indent=2) + "\n")
    print(json.dumps(d, indent=2))
    if d["decision"] == "INVALID_AUDIT_ERROR":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
