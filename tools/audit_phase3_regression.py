from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "phase3" / "regression_external_validity"
PROTOCOL = BASE / "CONFIRMATORY_PROTOCOL.json"
CONFIRM = BASE / "confirmatory"
SUMMARY = CONFIRM / "summary.json"


def exact_sign_p(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n)


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> None:
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert p["status"] == "LOCKED_BEFORE_CONFIRMATORY_OUTCOMES"
    seeds = list(range(2200, 2208))
    assert p["confirmatory_seeds"] == seeds
    assert p["exploration_seeds_excluded_from_inference"] == [2100, 2101, 2102]
    assert p["selection_rule"]["span_nmse_lte"] == 0.12
    assert p["selection_rule"]["val_r2_gte_teacher_minus"] == 0.05
    grid = [2, 4, 6, 8, 12, 16, 20, 24, 32]
    assert p["replacement"]["h_grid"] == grid

    runs = []
    for seed in seeds:
        r = json.loads((CONFIRM / f"seed_{seed}.json").read_text(encoding="utf-8"))
        runs.append(r)
        assert r["seed"] == seed
        assert r["dataset"] == "sklearn.datasets.load_diabetes"
        assert r["task"] == "tabular_regression"
        assert r["budget_grid_h"] == grid
        assert r["learned_parameter_budget_formula"] == "256*h for both conditions"
        assert r["selection_rule"]["span_nmse_lte"] == 0.12
        assert r["selection_rule"]["val_r2_gte_teacher_minus"] == 0.05
        assert r["selection_rule"]["test_used_for_selection"] is False
        assert r["data"]["feature_normalization"] == "train_mean_std_only"
        assert r["data"]["target_normalization"] == "train_mean_std_only"

        sep_first = None
        comp_first = None
        sep_test_rows = []
        comp_test_rows = []
        for row in r["grid"]:
            assert row["budget"] == 256 * row["h"]
            sep_expected = (
                row["sep_nmse"] <= 0.12
                and row["sep_val_r2"] >= r["teacher_val_r2"] - 0.05
            )
            comp_expected = (
                row["comp_nmse"] <= 0.12
                and row["comp_val_r2"] >= r["teacher_val_r2"] - 0.05
            )
            assert row["sep_pass"] is sep_expected
            assert row["comp_pass"] is comp_expected
            if sep_expected and sep_first is None:
                sep_first = row
            if comp_expected and comp_first is None:
                comp_first = row
            if "sep_test_r2" in row:
                sep_test_rows.append(row)
            if "comp_test_r2" in row:
                comp_test_rows.append(row)

        assert sep_first is not None and comp_first is not None
        assert r["selected_sep"] is not None and r["selected_comp"] is not None
        assert r["selected_sep"]["budget"] == sep_first["budget"]
        assert r["selected_comp"]["budget"] == comp_first["budget"]
        assert len(sep_test_rows) == 1 and sep_test_rows[0]["budget"] == sep_first["budget"]
        assert len(comp_test_rows) == 1 and comp_test_rows[0]["budget"] == comp_first["budget"]

        expected_log2 = math.log2(comp_first["budget"] / sep_first["budget"])
        assert close(r["log2_budget_ratio"], expected_log2)
        expected_test_diff = comp_first["comp_test_r2"] - sep_first["sep_test_r2"]
        assert close(r["test_r2_diff_comp_minus_sep"], expected_test_diff)

    assert s["confirmatory_seeds"] == seeds
    assert s["n_attempted"] == 8
    assert s["n_comparable"] == 8
    assert s["missing_endpoints"] == []

    vals = np.array([float(r["log2_budget_ratio"]) for r in runs])
    wins = int(np.sum(vals < 0))
    ties = int(np.sum(vals == 0))
    losses = int(np.sum(vals > 0))
    mean_log2 = float(vals.mean())
    geom = float(2.0**mean_log2)
    rng = np.random.default_rng(20260826)
    boots = np.array(
        [vals[rng.integers(0, len(vals), size=len(vals))].mean() for _ in range(10000)]
    )
    lo, hi = np.quantile(boots, [0.025, 0.975])

    assert s["wins_ties_losses"] == {"wins": wins, "ties": ties, "losses": losses}
    assert close(s["mean_log2_budget_ratio"], mean_log2)
    assert close(s["geometric_mean_budget_ratio"], geom)
    assert close(s["bootstrap95_mean_log2_budget_ratio"][0], lo)
    assert close(s["bootstrap95_mean_log2_budget_ratio"][1], hi)
    assert close(s["one_sided_exact_sign_p"], exact_sign_p(wins, losses))

    criteria = {
        "all_8_comparable": True,
        "composed_lower_at_least_7_of_8": wins >= 7,
        "bootstrap95_upper_below_zero": float(hi) < 0,
    }
    assert s["criteria"] == criteria
    assert s["status"] == ("VALID_PASS" if all(criteria.values()) else "VALID_FAIL")
    print("PHASE3 REGRESSION EVIDENCE AUDIT PASS")


if __name__ == "__main__":
    main()
