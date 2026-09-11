#!/usr/bin/env python3
"""Audit the reviewed publication candidate without clearing historical Issue #87."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str):
    return json.loads((ROOT / path).read_text())


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def git_blob_sha1(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def close_tree(actual, expected, path="root", atol=1e-12):
    errors = []
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            errors.append(f"{path}: {actual!r} != {expected!r}")
        return errors
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return [f"{path}: numeric type mismatch"]
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
            errors.append(f"{path}: {actual!r} != {expected!r}")
        return errors
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [f"{path}: list shape mismatch"]
        for i, (a, e) in enumerate(zip(actual, expected)):
            errors += close_tree(a, e, f"{path}[{i}]", atol)
        return errors
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: dict type mismatch"]
        if set(actual) != set(expected):
            errors.append(f"{path}: key mismatch actual={sorted(actual)} expected={sorted(expected)}")
            return errors
        for key in expected:
            errors += close_tree(actual[key], expected[key], f"{path}.{key}", atol)
        return errors
    if actual != expected:
        errors.append(f"{path}: {actual!r} != {expected!r}")
    return errors


def decision_map(ledger):
    return {x["id"]: x for x in ledger["decisions"]}


def audit():
    errors = []

    # Active publication integrity must pass, while the separate historical
    # 1200-1207 reproduction status remains explicitly unresolved.
    publication = load_module("publication_audit", "tools/audit_publication.py")
    publication_result = publication.audit(ROOT, False)
    if publication_result.get("integrity_status") != "PASS":
        errors.append("publication integrity audit failed: " + repr(publication_result.get("errors")))
    if publication_result.get("historical_reproduction_status") != "UNRESOLVED_ISSUE_87":
        errors.append("historical Issue #87 status drifted")
    if publication_result.get("historical_reproduction_blocks_current_r87r2_headline") is not False:
        errors.append("historical Issue #87 incorrectly blocks reviewed R87R2 headline")

    policy = load_json("publication/CANDIDATE_CLAIM_POLICY.json")
    ledger = load_json("publication/POST_V02_CLAIM_LEDGER.json")
    manifest = load_json("publication/CANDIDATE_EVIDENCE_MANIFEST.json")

    if policy.get("mode") != "REVIEWED_PUBLICATION_CANDIDATE_NOT_RELEASED":
        errors.append("candidate policy mode changed")
    if policy.get("review_base_main") != "c7fd4fb701064c9aeef5cc8109231d884e5258aa":
        errors.append("candidate review base changed")
    hist = policy.get("historical_archive", {})
    if hist.get("exact_reproduction_established") is not False or hist.get("issue") != 87 or hist.get("issue_remains_open") is not True or hist.get("r87r2_is_historical_recovery") is not False:
        errors.append("historical archive boundary weakened")
    limits = policy.get("claim_limits", {})
    if any(limits.get(k) is not False for k in [
        "independent_external_reproduction_claimed",
        "external_peer_review_claimed",
        "architecture_universality_claimed",
        "runtime_speedup_claimed",
        "ram_vram_energy_claimed",
        "llm_claimed",
        "universal_complexity_theorem_claimed",
    ]):
        errors.append("candidate contains an unsupported broad claim")
    release = policy.get("release_state", {})
    if any(release.get(k) is not False for k in ["merged_to_main", "release_tag_created", "announcement_posted", "issue_13_closed"]):
        errors.append("candidate incorrectly claims release/merge completion")

    if ledger.get("overall_decision") != "PASS_WITH_PUBLIC_SURFACE_EDITS_REQUIRED" or ledger.get("audit_status") != "PASS":
        errors.append("independent review ledger is not in reviewed PASS state")
    dm = decision_map(ledger)
    required_review = {
        "R87R2_FRESH_SQRT_PROFILE_CONFIRMATION": "KEEP",
        "PHASE4_CALIFORNIA_CONFIRMATORY": "KEEP",
        "PHASE3B_STRONGER_TEACHER_CONFIRMATORY": "EXCLUDE",
        "PHASE3C_NESTED_CV_TEACHER_SELECTION": "KEEP",
        "PHASE2E": "INVALIDATE",
    }
    for key, expected in required_review.items():
        if dm.get(key, {}).get("decision") != expected:
            errors.append(f"review disposition drift: {key}")

    # Vendored publication evidence must remain byte-identical to the reviewed
    # research commits; Git blob IDs are content identities, not branch refs.
    for name, expected_sha in manifest.get("git_blob_sha1", {}).items():
        p = ROOT / name
        if not p.exists():
            errors.append("missing vendored evidence: " + name)
            continue
        actual_sha = git_blob_sha1(p)
        if actual_sha != expected_sha:
            errors.append(f"vendored blob drift: {name}: {actual_sha} != {expected_sha}")

    # R87R2: fresh confirmatory direct baseline, explicitly not archive recovery.
    r87_protocol = load_json("results/reproduction/r87r2_fresh_sqrt_profile/PROTOCOL.json")
    r87_rows = load_json("results/reproduction/r87r2_fresh_sqrt_profile/FRESH_ROWS.json")
    r87_saved = load_json("results/reproduction/r87r2_fresh_sqrt_profile/DECISION.json")
    if r87_protocol.get("experiment") != "R87R2_FRESH_SQRT_PROFILE_CONFIRMATION":
        errors.append("R87R2 protocol identity changed")
    if r87_protocol.get("evidence_class") != "PROSPECTIVE_CONFIRMATORY_CONDITIONAL_ON_PROFILE_GATE":
        errors.append("R87R2 protocol evidence class changed")
    if r87_protocol.get("fresh_seeds") != list(range(871200, 871216)):
        errors.append("R87R2 fresh cohort changed")
    if r87_protocol.get("T", {}).get("scientific_runner_blob") != "8759933ed2bb95014c3afc835acafa1743e40ea6":
        errors.append("R87R2 scientific runner changed")
    if r87_protocol.get("T", {}).get("bootstrap_seed") != 87122026 or r87_protocol.get("T", {}).get("bootstrap_resamples") != 100000:
        errors.append("R87R2 bootstrap lock changed")
    historical_rule = r87_protocol.get("D", {}).get("historical", "")
    if "not old-value recovery" not in historical_rule or "Issue87 closure" not in historical_rule:
        errors.append("R87R2 historical boundary text missing")
    if r87_rows.get("primary_host") != "a":
        errors.append("R87R2 primary host changed")
    r87_seed_rows = r87_rows.get("rows", [])
    if sorted(x.get("seed") for x in r87_seed_rows) != list(range(871200, 871216)):
        errors.append("R87R2 raw cohort incomplete or changed")
    try:
        r87_eval = load_module("r87r2_confirm_eval", "scripts/reproduction_diagnostics/r87r2_confirm.py")
        r87_recalc = r87_eval.evaluate(r87_seed_rows)
        errors += ["R87R2 recalculation: " + x for x in close_tree(r87_recalc, r87_saved)]
    except Exception as exc:
        errors.append("R87R2 recalculation exception: " + repr(exc))
    if r87_saved.get("decision") != "R87R2_CONFIRMATORY_PASS" or r87_saved.get("historical_archive_recovered") is not False or r87_saved.get("test_used_for_selection") is not False:
        errors.append("R87R2 locked decision/boundary changed")

    # Phase4: one California Housing split, residual-MLP family, prospective
    # Stage-A teacher selection followed by fresh Stage-B confirmatory seeds.
    p4_stage_a = load_json("results/phase4/california_regression/STAGE_A_RESULT.json")
    p4_protocol = load_json("results/phase4/california_regression/STAGE_B_CONFIRMATORY_PROTOCOL.json")
    p4_rows = load_json("results/phase4/california_regression/stage_b_confirmatory/FRESH_ROWS.json")
    p4_saved = load_json("results/phase4/california_regression/stage_b_confirmatory/DECISION.json")
    p4_technical = load_json("results/phase4/california_regression/stage_b_confirmatory/TECHNICAL_REPLICATION.json")
    if p4_stage_a.get("stage_a_status") != "PASS_SELECT_RECIPE" or p4_stage_a.get("test_evaluated") is not False or p4_stage_a.get("chosen_recipe", {}).get("id") != "low_lr_30":
        errors.append("Phase4 Stage A selection boundary changed")
    if p4_protocol.get("status") != "LOCKED_BEFORE_STAGE_B_OUTCOMES" or p4_protocol.get("evidence_class") != "PROSPECTIVE_CONFIRMATORY":
        errors.append("Phase4 Stage B lock changed")
    if p4_protocol.get("fresh_model_seeds") != list(range(2900, 2908)):
        errors.append("Phase4 fresh cohort changed")
    if p4_protocol.get("replacement", {}).get("test_used_for_selection") is not False:
        errors.append("Phase4 test-selection boundary changed")
    ds = p4_protocol.get("dataset", {})
    if ds.get("X_float32_sha256") != "ef53061b11adc508abf7f3d57bbeb8a90302caedaa3201b846cdd5f65caddfd5" or ds.get("y_float32_sha256") != "4fcec16744c2835e99b638e9ac95cde0619da1673dfdf7e638d31cd83f32fd14":
        errors.append("Phase4 dataset identity changed")
    p4_seed_rows = p4_rows.get("rows", [])
    if p4_rows.get("primary_host") != "a" or sorted(x.get("seed") for x in p4_seed_rows) != list(range(2900, 2908)):
        errors.append("Phase4 primary raw cohort incomplete or changed")
    try:
        p4_eval = load_module("phase4_confirm_eval", "scripts/phase4/evaluate_california_confirm.py")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            inputs = []
            for row in p4_seed_rows:
                p = td / f"seed_{row['seed']}.json"
                p.write_text(json.dumps(row))
                inputs.append(str(p))
            out = td / "decision.json"
            rc = p4_eval.main(inputs, str(out))
            if rc != 0:
                errors.append(f"Phase4 evaluator returned {rc}")
            p4_recalc = json.loads(out.read_text())
        errors += ["Phase4 recalculation: " + x for x in close_tree(p4_recalc, p4_saved)]
    except Exception as exc:
        errors.append("Phase4 recalculation exception: " + repr(exc))
    if p4_saved.get("decision") != "PHASE4_CONFIRMATORY_PASS":
        errors.append("Phase4 locked decision changed")
    if p4_technical.get("status") != "PASS" or p4_technical.get("decision_exact") is not True or len(p4_technical.get("pairs", [])) != 8 or not all(x.get("scientific_record_exact") is True for x in p4_technical.get("pairs", [])):
        errors.append("Phase4 technical replication is not exact PASS")

    # Reviewed headline metrics must agree with vendored decision objects.
    try:
        rmetric = dm["R87R2_FRESH_SQRT_PROFILE_CONFIRMATION"]["reviewed_metrics"]
        pmetric = dm["PHASE4_CALIFORNIA_CONFIRMATORY"]["reviewed_metrics"]
        if rmetric["n"] != 16 or not math.isclose(rmetric["geometric_composed_over_component_budget_ratio"], r87_saved["geometric_budget_ratio"], rel_tol=0.0, abs_tol=1e-15):
            errors.append("R87R2 review metrics disagree with vendored decision")
        if pmetric["n"] != 8 or not math.isclose(pmetric["geometric_composed_over_component_budget_ratio"], p4_saved["composition_budget"]["geometric_mean_ratio"], rel_tol=0.0, abs_tol=1e-15):
            errors.append("Phase4 review metrics disagree with vendored decision")
    except Exception as exc:
        errors.append("review metric cross-check exception: " + repr(exc))

    return {
        "candidate": policy.get("candidate"),
        "status": "FAIL" if errors else "PASS",
        "decision": "CANDIDATE_EVIDENCE_PASS" if not errors else "CANDIDATE_EVIDENCE_FAIL",
        "historical_issue_87_closed": False,
        "historical_archive_exactly_reproduced": False,
        "r87r2_current_direct_baseline": r87_saved.get("decision"),
        "phase4_bounded_external_validity": p4_saved.get("decision"),
        "vendored_blob_count": len(manifest.get("git_blob_sha1", {})),
        "errors": errors,
        "scope": "Reviewed candidate evidence integrity and raw-row recalculation. Not external peer review, independent external reproduction, merge approval, release, or historical archive recovery."
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = audit()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
