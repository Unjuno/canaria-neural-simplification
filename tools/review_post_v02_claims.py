#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np


def git(*args: str, check: bool = True) -> str:
    p = subprocess.run(["git", *args], text=True, capture_output=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr}")
    return p.stdout.strip()


def load_json(ref: str, path: str):
    return json.loads(git("show", f"{ref}:{path}"))


def first_add(ref: str, path: str) -> str:
    out = git("log", "--reverse", "--diff-filter=A", "--format=%H", ref, "--", path)
    vals = [x for x in out.splitlines() if x]
    if not vals:
        raise AssertionError(f"no add commit for {ref}:{path}")
    return vals[0]


def assert_ancestor(older: str, newer: str, label: str):
    p = subprocess.run(["git", "merge-base", "--is-ancestor", older, newer])
    if p.returncode != 0:
        raise AssertionError(f"chronology failure {label}: {older} !< {newer}")


def close(a: float, b: float, tol: float = 1e-12):
    if not math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"float mismatch {a!r} != {b!r}")


def close_pair(a, b, tol: float = 1e-12):
    assert len(a) == len(b) == 2
    close(a[0], b[0], tol)
    close(a[1], b[1], tol)


def bootstrap_shared(arrays: dict[str, np.ndarray], seed: int, n_boot: int):
    names = list(arrays)
    n = len(arrays[names[0]])
    if any(len(arrays[k]) != n for k in names):
        raise AssertionError("paired bootstrap arrays have unequal length")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for k, x in arrays.items():
        boots = x[idx].mean(axis=1)
        out[k] = {
            "mean": float(x.mean()),
            "ci95": [float(v) for v in np.percentile(boots, [2.5, 97.5])],
            "bootstrap_se": float(boots.std(ddof=1)),
        }
    return out


def chronology(ref: str, pairs: list[tuple[str, str, str]]):
    rows = []
    for label, protocol_path, outcome_path in pairs:
        p = first_add(ref, protocol_path)
        o = first_add(ref, outcome_path)
        assert p != o, f"{label}: protocol and outcome added in same commit"
        assert_ancestor(p, o, label)
        rows.append({"label": label, "protocol_add": p, "outcome_add": o, "status": "PASS"})
    return rows


def audit_r87r2():
    ref = "origin/research/r87-portable-profile-validation"
    base = "results/reproduction/r87r2_fresh_sqrt_profile"
    protocol = load_json(ref, f"{base}/PROTOCOL.json")
    raw = load_json(ref, f"{base}/FRESH_ROWS.json")
    dec = load_json(ref, f"{base}/DECISION.json")
    rows = raw["rows"]
    seeds = [r["seed"] for r in rows]
    assert seeds == list(range(871200, 871216))
    assert len(set(seeds)) == 16
    assert all(r["selected_sep"] is not None and r["selected_comp"] is not None for r in rows)
    assert protocol["fresh_seeds"] == seeds
    assert "No test metric feeds selection" in protocol["T"]["test_boundary"]

    logs = np.array([math.log2(r["selected_comp"]["budget"] / r["selected_sep"]["budget"]) for r in rows])
    diffs = np.array([r["selected_comp"]["comp_test_acc"] - r["selected_sep"]["sep_test_acc"] for r in rows])
    b = bootstrap_shared({"log": logs, "diff": diffs}, dec["bootstrap_seed"], dec["bootstrap_resamples"])
    close(b["log"]["mean"], dec["primary_log2_budget_ratio"]["mean"])
    close_pair(b["log"]["ci95"], dec["primary_log2_budget_ratio"]["ci95"])
    close(b["diff"]["mean"], dec["secondary_test_accuracy_difference"]["mean"])
    close_pair(b["diff"]["ci95"], dec["secondary_test_accuracy_difference"]["ci95"])
    geom = float(2.0 ** logs.mean())
    close(geom, dec["geometric_budget_ratio"])
    composed_lower = sum(r["selected_comp"]["budget"] < r["selected_sep"]["budget"] for r in rows)
    assert composed_lower == dec["composed_lower_count"] == 15
    assert dec["primary_log2_budget_ratio"]["ci95"][1] < 0
    assert dec["secondary_test_accuracy_difference"]["ci95"][0] > -0.02
    assert dec["decision"] == "R87R2_CONFIRMATORY_PASS"
    assert dec["historical_archive_recovered"] is False
    return {
        "status": "KEEP_BOUNDED_NEW_BASELINE",
        "decision": dec["decision"],
        "n": 16,
        "geometric_budget_ratio": geom,
        "log2_ci95": b["log"]["ci95"],
        "selected_test_accuracy_difference_ci95": b["diff"]["ci95"],
        "historical_archive_recovered": False,
        "chronology": chronology(ref, [("R87R2", f"{base}/PROTOCOL.json", f"{base}/FRESH_ROWS.json")]),
    }


def audit_phase3b():
    ref = "origin/research/phase3b-confirmatory"
    pbase = "results/phase3b/stronger_teacher_regression"
    base = f"{pbase}/stage_b_confirmatory"
    protocol = load_json(ref, f"{pbase}/STAGE_B_CONFIRMATORY_PROTOCOL.json")
    raw = load_json(ref, f"{base}/FRESH_ROWS.json")
    dec = load_json(ref, f"{base}/DECISION.json")
    rows = raw["rows"]
    seeds = [r["seed"] for r in rows]
    assert seeds == list(range(2400, 2408))
    assert protocol["fresh_model_seeds"] == seeds
    assert all(r["selection_rule"]["test_used_for_selection"] is False for r in rows)

    cand = np.array([r["candidate_teacher_test_r2"] for r in rows])
    gain = np.array([r["candidate_minus_baseline_test_r2"] for r in rows])
    logs = np.array([math.log2(r["selected_comp"]["budget"] / r["selected_sep"]["budget"]) for r in rows])
    util = np.array([r["selected_comp"]["comp_test_r2"] - r["selected_sep"]["sep_test_r2"] for r in rows])
    b = bootstrap_shared({"cand": cand, "gain": gain, "log": logs, "util": util}, dec["bootstrap_seed"], dec["bootstrap_resamples"])
    close_pair(b["cand"]["ci95"], dec["teacher_absolute"]["ci95"])
    close_pair(b["gain"]["ci95"], dec["teacher_improvement"]["ci95"])
    close_pair(b["log"]["ci95"], dec["composition_budget"]["ci95"])
    close_pair(b["util"]["ci95"], dec["selected_utility"]["ci95"])
    assert dec["teacher_absolute"]["status"] == "UNCERTAIN"
    assert dec["teacher_improvement"]["status"] == "UNCERTAIN"
    assert dec["composition_budget"]["status"] == "PASS"
    assert dec["selected_utility"]["status"] == "PASS"
    assert dec["decision"] == "PHASE3B_CONFIRMATORY_UNCERTAIN"
    return {
        "status": "EXCLUDE_POSITIVE_EXTERNAL_VALIDITY_KEEP_BOUNDARY",
        "decision": dec["decision"],
        "teacher_absolute": dec["teacher_absolute"],
        "teacher_improvement": dec["teacher_improvement"],
        "composition_budget": dec["composition_budget"],
        "selected_utility": dec["selected_utility"],
        "chronology": chronology(ref, [("Phase3B", f"{pbase}/STAGE_B_CONFIRMATORY_PROTOCOL.json", f"{base}/FRESH_ROWS.json")]),
    }


def audit_phase3c():
    ref = "origin/research/phase3c-nested-cv-regression"
    base = "results/phase3c/nested_cv_teacher"
    result = load_json(ref, f"{base}/RESULT.json")
    manifest = load_json(ref, f"{base}/RUN_MANIFEST.json")
    assert result["outer_test_evaluated"] is False
    assert result["chosen_recipe"] is None
    assert not any(x["eligible"] for x in result["recipe_aggregates"])
    best = max(result["recipe_aggregates"], key=lambda x: x["mean_validation_r2"])
    assert best["recipe"]["id"] == "short_15"
    close(best["mean_validation_r2"], 0.33593505423120684)
    assert manifest["rows"] == 195
    assert manifest["technical_replication"] == "EXACT_PASS"
    prot_add = first_add(ref, f"{base}/PROTOCOL.json")
    amend_add = first_add(ref, f"{base}/PRE_OUTCOME_AMENDMENT.json")
    result_add = first_add(ref, f"{base}/RESULT.json")
    assert_ancestor(prot_add, amend_add, "Phase3C protocol->amendment")
    assert_ancestor(amend_add, result_add, "Phase3C amendment->result")
    return {
        "status": "KEEP_NEGATIVE_BOUNDARY",
        "outer_test_evaluated": False,
        "chosen_recipe": None,
        "best_recipe": best["recipe"]["id"],
        "best_mean_validation_r2": best["mean_validation_r2"],
        "technical_replication": manifest["technical_replication"],
        "chronology": {"protocol_add": prot_add, "amendment_add": amend_add, "result_add": result_add, "status": "PASS"},
    }


def audit_phase4():
    ref = "origin/research/phase4-california-regression"
    pbase = "results/phase4/california_regression"
    base = f"{pbase}/stage_b_confirmatory"
    stage_a = load_json(ref, f"{pbase}/STAGE_A_RESULT.json")
    protocol = load_json(ref, f"{pbase}/STAGE_B_CONFIRMATORY_PROTOCOL.json")
    raw = load_json(ref, f"{base}/FRESH_ROWS.json")
    dec = load_json(ref, f"{base}/DECISION.json")
    technical = load_json(ref, f"{base}/TECHNICAL_REPLICATION.json")
    rows = raw["rows"]
    seeds = [r["seed"] for r in rows]
    assert stage_a["chosen_recipe"]["id"] == "low_lr_30"
    assert protocol["fresh_model_seeds"] == seeds == list(range(2900, 2908))
    assert all(r["selection_rule"]["test_used_for_selection"] is False for r in rows)

    teacher = np.array([r["teacher_test_r2"] for r in rows])
    logs = np.array([math.log2(r["selected_comp"]["budget"] / r["selected_sep"]["budget"]) for r in rows])
    util = np.array([r["selected_comp"]["comp_test_r2"] - r["selected_sep"]["sep_test_r2"] for r in rows])
    b = bootstrap_shared({"teacher": teacher, "log": logs, "util": util}, dec["bootstrap_seed"], dec["bootstrap_resamples"])
    close_pair(b["teacher"]["ci95"], dec["teacher_quality"]["ci95"])
    close_pair(b["log"]["ci95"], dec["composition_budget"]["ci95"])
    close_pair(b["util"]["ci95"], dec["selected_utility"]["ci95"])
    geom = float(2.0 ** logs.mean())
    close(geom, dec["composition_budget"]["geometric_mean_ratio"])
    assert dec["teacher_quality"]["ci95"][0] > 0.70
    assert dec["composition_budget"]["ci95"][1] < 0
    assert dec["composition_budget"]["composed_lower_count"] == 8
    assert dec["selected_utility"]["ci95"][0] > -0.03
    assert dec["decision"] == "PHASE4_CONFIRMATORY_PASS"
    assert technical.get("status") == "PASS" or technical.get("technical_replication") == "PASS"
    return {
        "status": "KEEP_BOUNDED_EXTERNAL_VALIDITY",
        "decision": dec["decision"],
        "n": 8,
        "teacher_test_r2_ci95": b["teacher"]["ci95"],
        "geometric_budget_ratio": geom,
        "log2_budget_ratio_ci95": b["log"]["ci95"],
        "selected_test_r2_difference_ci95": b["util"]["ci95"],
        "chronology": chronology(ref, [
            ("Phase4 StageA", f"{pbase}/STAGE_A_PROTOCOL.json", f"{pbase}/STAGE_A_RESULT.json"),
            ("Phase4 StageB", f"{pbase}/STAGE_B_CONFIRMATORY_PROTOCOL.json", f"{base}/FRESH_ROWS.json"),
        ]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = {
        "review": "POST_V02_INDEPENDENT_CLAIM_REVIEW",
        "main_review_base": "c7fd4fb701064c9aeef5cc8109231d884e5258aa",
        "note": "Independent recomputation from persisted raw rows and git chronology; not external peer review.",
        "R87R2": audit_r87r2(),
        "PHASE3B": audit_phase3b(),
        "PHASE3C": audit_phase3c(),
        "PHASE4": audit_phase4(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v.get("status") for k, v in report.items() if isinstance(v, dict)}))


if __name__ == "__main__":
    main()
