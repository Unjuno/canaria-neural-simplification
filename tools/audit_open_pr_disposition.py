#!/usr/bin/env python3
"""Audit explicit publication dispositions for every open pull request in the snapshot."""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "Unjuno/canaria-neural-simplification"
DISPOSITION_PATH = ROOT / "publication/OPEN_PR_DISPOSITION_2026-09-12.json"


def live_open_pr_numbers() -> set[int]:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required for --live-github")
    out: set[int] = set()
    page = 1
    while True:
        url = f"https://api.github.com/repos/{REPO}/pulls?state=open&per_page=100&page={page}"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "canaria-publication-open-pr-audit",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            rows = json.load(resp)
        out.update(int(row["number"]) for row in rows)
        if len(rows) < 100:
            break
        page += 1
    return out


def audit(live_github: bool) -> dict:
    d = json.loads(DISPOSITION_PATH.read_text())
    errors: list[str] = []

    if d.get("schema_version") != 1 or d.get("review") != "OPEN_PR_PUBLICATION_DISPOSITION_2026_09_12":
        errors.append("unexpected open-PR disposition schema/review")
    if d.get("review_base_main") != "c7fd4fb701064c9aeef5cc8109231d884e5258aa":
        errors.append("open-PR review base changed")

    snapshot = d.get("open_pr_snapshot", {})
    numbers = snapshot.get("numbers", [])
    if snapshot.get("count") != len(numbers):
        errors.append("open-PR snapshot count mismatch")
    if len(numbers) != len(set(numbers)):
        errors.append("duplicate PR number in open-PR snapshot")

    rows = d.get("prs", [])
    by_pr = {row.get("pr"): row for row in rows}
    if len(rows) != len(by_pr):
        errors.append("duplicate PR disposition row")
    if set(numbers) != set(by_pr):
        errors.append("snapshot numbers and disposition rows differ")

    for number, row in by_pr.items():
        if row.get("mechanical_merge") is not False:
            errors.append(f"PR #{number} permits mechanical merge")
        if not row.get("disposition") or "UNDECIDED" in row.get("disposition", ""):
            errors.append(f"PR #{number} has no explicit disposition")
        if not row.get("claim_role"):
            errors.append(f"PR #{number} has no claim role")

    expected = {
        11: "EXCLUDE_FROM_CURRENT_HEADLINE_KEEP_RESEARCH",
        14: "SUPERSEDED_GATE_LOGIC_DO_NOT_MECHANICALLY_MERGE",
        17: "STRUCTURAL_HISTORY_DO_NOT_MECHANICALLY_MERGE",
        56: "SELECTIVE_SUPPORT_ONLY_NO_WHOLESALE_MERGE",
        68: "EXCLUDE_HEADLINE_KEEP_PROVENANCE",
        89: "HISTORICAL_PROVENANCE_DIAGNOSTIC_ONLY",
        90: "SELECTIVE_VENDOR_KEEP_PRIMARY_BASELINE",
        91: "EXCLUDE_POSITIVE_HEADLINE_KEEP_BOUNDARY",
        93: "KEEP_NEGATIVE_BOUNDARY_EVIDENCE",
        95: "SELECTIVE_VENDOR_KEEP_BOUNDED_SUPPORT",
        100: "REVIEW_PREREQUISITE_FOR_PUBLICATION_CANDIDATE",
        101: "FINAL_PUBLICATION_REVIEW_REQUIRED",
    }
    for number in range(69, 87):
        expected[number] = "RESEARCH_APPENDIX_ONLY"
    for number, disposition in expected.items():
        if by_pr.get(number, {}).get("disposition") != disposition:
            errors.append(f"PR #{number} disposition drift")

    extra = d.get("additional_review", {})
    if extra.get("pr") != 11 or extra.get("decision") != "EXCLUDE_FROM_CURRENT_HEADLINE_KEEP_AS_VALID_BOUNDED_RESEARCH_RESULT":
        errors.append("PR #11 publication review decision changed")
    recalc = extra.get("recalculation", {})
    if recalc.get("matches_persisted_summary") is not True:
        errors.append("PR #11 independent recalculation not marked exact")
    if recalc.get("bootstrap_resamples") != 10000 or recalc.get("bootstrap_rng_seed") != 20260826:
        errors.append("PR #11 locked bootstrap definition changed")
    if recalc.get("mean_log2_budget_ratio") != -1.0263620978123273:
        errors.append("PR #11 recalculated mean changed")
    if recalc.get("bootstrap95") != [-1.1842413985415514, -0.868482797083103]:
        errors.append("PR #11 recalculated CI changed")
    limits = extra.get("limitations", {})
    if limits.get("teacher_eligibility_filter_preregistered") is not False:
        errors.append("PR #11 teacher-eligibility limitation changed")
    if limits.get("componentwise_grid_ceiling_seeds") != [2201, 2204]:
        errors.append("PR #11 grid-ceiling limitation changed")
    if limits.get("selected_test_utility_improvement_claim_supported") is not False:
        errors.append("PR #11 test-utility boundary changed")

    policy = d.get("policy", {})
    if policy.get("mechanical_research_pr_merges_allowed") is not False:
        errors.append("mechanical research-PR merge policy weakened")
    if policy.get("new_open_scientific_prs_after_this_snapshot_require_explicit_review_before_announcement") is not True:
        errors.append("new-open-PR review requirement weakened")
    if policy.get("ci_pass_is_not_merge_approval") is not True:
        errors.append("CI/merge boundary weakened")

    gate = d.get("announcement_gate", {})
    if gate.get("snapshot_has_unclassified_open_prs") is not False:
        errors.append("snapshot still marks unclassified PRs")
    for key in ["issue13_can_close_now", "main_merge_authorized", "announcement_authorized"]:
        if gate.get(key) is not False:
            errors.append(f"announcement gate unexpectedly authorizes {key}")
    for key in ["independent_review_of_pr100_still_required", "independent_review_of_pr101_still_required"]:
        if gate.get(key) is not True:
            errors.append(f"independent-review prerequisite weakened: {key}")

    live_numbers: list[int] | None = None
    unclassified_live: list[int] = []
    if live_github:
        try:
            live = live_open_pr_numbers()
            live_numbers = sorted(live)
            unclassified_live = sorted(live - set(by_pr))
            if unclassified_live:
                errors.append("live open PRs lack publication disposition: " + ",".join(map(str, unclassified_live)))
        except Exception as exc:
            errors.append("live GitHub open-PR audit failed: " + repr(exc))

    return {
        "status": "FAIL" if errors else "PASS",
        "decision": "OPEN_PR_DISPOSITION_PASS" if not errors else "OPEN_PR_DISPOSITION_FAIL",
        "snapshot_count": len(numbers),
        "classified_count": len(by_pr),
        "live_github_checked": live_github,
        "live_open_pr_numbers": live_numbers,
        "unclassified_live_open_prs": unclassified_live,
        "errors": errors,
        "scope": "Publication inclusion/exclusion governance. This does not approve merges, close PRs, provide external peer review, or convert research-only evidence into headline evidence.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--live-github", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    result = audit(args.live_github)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
