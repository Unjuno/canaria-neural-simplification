#!/usr/bin/env python3
"""Integrity audit for the imported C59-C61 Gaussian-shift interface line.

This audit verifies that repository metadata does not silently over-promote the
imported handoff and that the transcribed aggregate values satisfy the stated
C59/C60 gates. It does not convert imported aggregates into reconstructed raw
evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "gaussian_shift_interface"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def audit_handoff() -> None:
    data = load(RESULTS / "HANDOFF_C59_C60.json")
    require(data["provenance_class"] == "IMPORTED_HANDOFF_RESULT", "C59/C60 provenance drift")
    require(data["scope"]["shift"] == "Gaussian", "shift family drift")
    require(float(data["scope"]["sigma"]) == 0.04, "sigma drift")
    require(data["scope"]["architecture"] == "Residual CNN", "architecture drift")

    c59 = data["c59"]
    require(c59["status"] == "P8_NONINFERIOR_PASS", "C59 decision drift")
    require(c59["eligible"] == 16 and c59["attempted"] == 16, "C59 cohort drift")
    require(len(c59["fresh_seeds"]) == 16, "C59 seed count drift")
    require(c59["fresh_seeds"] == list(range(47400, 47416)), "C59 seed identity drift")
    require(c59["validation_accuracy_difference_bootstrap95_pp"][0] > c59["validation_noninferiority_margin_pp"], "C59 validation NI gate no longer passes")
    require(c59["nmse_geomean_ratio_bootstrap95"][1] < c59["nmse_ratio_margin"], "C59 NMSE-ratio gate no longer passes")
    require(c59["p8_delta_nmse_vs_frozen_bootstrap95"][1] < 0.0, "C59 frozen-improvement gate no longer passes")
    require(c59["teacher_shift_safeguard"] == "PASS", "C59 teacher safeguard drift")
    require(c59["test_used"] is False, "C59 must remain test-unused")
    require(c59["parameter_match"] == "PASS", "C59 parameter-match drift")
    require(c59["independent_audit_reported"] == "PASS", "C59 imported audit status drift")

    c60 = data["c60"]
    require(c60["status"] == "ADVANCE_P4_TO_C61", "C60 decision drift")
    require(c60["evidence_class"] == "exploratory", "C60 evidence class drift")
    require(c60["eligible"] == 16 and c60["attempted"] == 16, "C60 cohort drift")
    require(c60["nmse_geomean_ratio_bootstrap95"][1] < 1.25, "C60 imported NMSE advance signal drift")
    require(c60["p4_delta_nmse_vs_frozen_bootstrap95"][1] < 0.0, "C60 imported frozen-improvement signal drift")
    require(c60["advance_gates"] == "PASS", "C60 imported advance status drift")
    require(c60["confirmatory_claim_allowed"] is False, "C60 must not become confirmatory")


def audit_c61_protocol() -> None:
    data = load(RESULTS / "c61" / "IMPORTED_PROTOCOL.json")
    require(data["experiment"] == "C61", "C61 identity drift")
    require(data["provenance_class"] == "IMPORTED_LOCKED_PROTOCOL", "C61 provenance drift")
    require(data["github_timestamp_is_preregistration"] is False, "C61 GitHub timestamp must not be represented as preregistration")
    require(data["fresh_seeds"] == list(range(49400, 49416)), "C61 seed set drift")
    require(data["comparison"] == {"candidate": "P4", "reference": "P8"}, "C61 comparison drift")
    require(data["calibration_samples"] == 192, "C61 calibration count drift")
    require(float(data["shift"]["sigma"]) == 0.04, "C61 sigma drift")
    gates = data["gates"]
    require(gates["minimum_eligible"] == 8, "C61 minimum-eligible drift")
    require(float(gates["validation_accuracy_noninferiority_margin_pp"]) == -2.0, "C61 validation margin drift")
    require(float(gates["nmse_geomean_ratio_margin"]) == 1.25, "C61 NMSE margin drift")
    require(gates["bootstrap_resamples"] == 100000, "C61 bootstrap count drift")
    require(gates["test_used"] is False, "C61 must remain test-unused")
    require(data["fresh_outcome_status_in_repository"] == "UNKNOWN", "C61 outcome was changed without updating audit policy")
    require(data["decision"] is None, "C61 decision must remain null before evidence import")
    eq = data["reported_equivalence_check"]
    require(eq["seed"] == 48400, "C61 equivalence seed drift")
    require(eq["max_reported_difference"] == 0.0, "C61 equivalence report drift")
    require(eq["scientific_outcome"] is False, "equivalence check must not be promoted to scientific outcome")
    require(len(data["required_missing_artifacts"]) >= 5, "C61 provenance gaps were silently erased")


def audit_template() -> None:
    data = load(RESULTS / "c61" / "SEED_ROWS_TEMPLATE.json")
    require(data["evidence_status"] == "TEMPLATE_NOT_EVIDENCE", "C61 template must remain non-evidence")
    require(data["required_complete_seed_set"] == list(range(49400, 49416)), "C61 template cohort drift")
    for row in data["rows"]:
        require(row["eligible"] is None, "template contains an eligibility outcome")
        for key in ("p4_validation_accuracy", "p8_validation_accuracy", "p4_nmse", "p8_nmse"):
            require(row[key] is None, f"template contains fabricated outcome value: {key}")


def main() -> None:
    audit_handoff()
    audit_c61_protocol()
    audit_template()
    print("GAUSSIAN_SHIFT_INTERFACE_AUDIT_PASS")


if __name__ == "__main__":
    main()
