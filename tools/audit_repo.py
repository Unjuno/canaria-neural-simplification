from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


# Parse every retained artifact, including archives. Historical code may be old,
# but checked-in Python/JSON/CSV must remain syntactically readable.
for p in ROOT.rglob("*.py"):
    if ".git" in p.parts:
        continue
    try:
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    except Exception as exc:
        errors.append(f"python syntax: {rel(p)}: {exc}")

for p in ROOT.rglob("*.json"):
    try:
        json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"json: {rel(p)}: {exc}")

for p in ROOT.rglob("*.csv"):
    try:
        with p.open(newline="", encoding="utf-8") as f:
            list(csv.reader(f))
    except Exception as exc:
        errors.append(f"csv: {rel(p)}: {exc}")

required_current = [
    "README.md",
    "QUICKSTART.md",
    "STATUS.md",
    "CHANGELOG.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "REPOSITORY_LAYOUT.md",
    "docs/README.md",
    "docs/ANNOUNCEMENT_READINESS.md",
    "docs/HISTORICAL_INDEX.md",
    "docs/CORE_DISCOVERY.md",
    "docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md",
    "docs/CORE_DISCOVERY_REPLICATION_DIGITS.md",
    "docs/CLAIMS_AND_EVIDENCE.md",
    "docs/INDEPENDENT_REREVIEW_2026-08-26.md",
    "docs/PUBLICATION_NOTES.md",
    "docs/TERMINOLOGY.md",
    "docs/FAQ.md",
    "docs/TRAINING_TIME_CONSOLIDATION.md",
    "docs/LATE_STAGE_FINDINGS.md",
    "docs/NEGATIVE_RESULTS.md",
    "docs/APPLICATIONS.md",
    "docs/RUNTIME_POC.md",
    "docs/OPEN_QUESTIONS.md",
    "docs/REPRODUCIBILITY.md",
    "docs/ROADMAP.md",
    "docs/phase2/README.md",
    "results/training_time/summary.json",
    "results/training_time/protocol_manifest.json",
    "results/training_time/late_stage_summary.json",
    "results/training_time/ARTIFACT_INVENTORY.md",
    "results/reproduction/README.md",
    "scripts/reproduce/g7_confirmatory/run_seed.py",
    "scripts/reproduce/g7_confirmatory/README.md",
    "scripts/reproduce/g7_confirmatory/requirements.txt",
    "scripts/reproduce/g7_confirmatory/runtime_poc.py",
    "results/reproduction/g7_seed4300_report.json",
    "results/reproduction/runtime_poc_seed4300_report.json",
    ".github/workflows/reproduce-g7.yml",
    ".github/workflows/runtime-poc.yml",
    "scripts/replication/vit_compositional.py",
    "results/replication/vit_compositional/PROTOCOL_LOCK.json",
    "results/replication/vit_compositional/confirm_summary.json",
    "results/replication/vit_compositional/seed_table.csv",
    "scripts/reproduce/core_discovery_digits/run_confirmatory.py",
    "results/core_discovery_digits/PROTOCOL_LOCK.json",
    "results/core_discovery_digits/confirm_summary.json",
    "results/phase2/precision_composition/CORRECTION_STATUS.json",
    "results/phase2/precision_composition/INVALIDATED_HISTORY.md",
]

required_provenance = [
    "archives/README.md",
    "archives/reviews/REVIEW_HANDOFF_2026-08-26.md",
    "archives/releases/v0.2.0/PUBLIC_SNAPSHOT.md",
    "archives/releases/v0.2.0/RELEASE_CHECKLIST.md",
    "archives/research-history/docs/history",
    "archives/research-history/docs/phases",
    "archives/research-history/results/history",
    "archives/research-history/results/phaseA_v11/stage3_confirmatory_summary.json",
    "archives/research-history/results/v25",
    "archives/research-history/scripts/phases/v23",
    "archives/research-history/environment/history/v10",
]

for item in required_current + required_provenance:
    p = ROOT / item
    if not p.exists():
        errors.append(f"missing required path: {item}")

# Check relative links only on the active public/current surface. Archived files
# intentionally preserve old text and may contain historical relative paths.
public_markdown = [ROOT / item for item in required_current if item.endswith(".md")]
md_link = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
for p in public_markdown:
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8")
    for raw_target in md_link.findall(text):
        target = raw_target.strip()
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = target.split()[0].strip("<>")
        target = unquote(target.split("#", 1)[0])
        if not target:
            continue
        candidate = (p.parent / target).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError:
            errors.append(f"current markdown link escapes repository: {rel(p)} -> {raw_target}")
            continue
        if not candidate.exists():
            errors.append(f"broken current markdown link: {rel(p)} -> {raw_target}")

# Supported public runners must not regain private working-directory dependencies.
for runner_root in (
    ROOT / "scripts/reproduce",
    ROOT / "scripts/replication",
    ROOT / "scripts/phase2",
):
    if runner_root.exists():
        for p in runner_root.rglob("*.py"):
            text = p.read_text(encoding="utf-8")
            if "/mnt/data" in text:
                errors.append(f"private /mnt/data dependency in public runner: {rel(p)}")

late_path = ROOT / "results/training_time/late_stage_summary.json"
if late_path.exists():
    late = json.loads(late_path.read_text(encoding="utf-8"))
    experiments = late.get("experiments", {})
    for name in ("G18", "G19", "G20d", "G20e", "G21", "G22", "G23", "G24", "G25", "G26"):
        if name not in experiments:
            errors.append(f"late-stage manifest missing experiment: {name}")
    if experiments.get("G21", {}).get("status") != "FAIL":
        errors.append("late-stage manifest must preserve G21 as FAIL")

repro_path = ROOT / "results/reproduction/g7_seed4300_report.json"
if repro_path.exists():
    repro = json.loads(repro_path.read_text(encoding="utf-8"))
    expected_hash = "68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028"
    if repro.get("status") != "PASS" or repro.get("exact_json_equal") is not True:
        errors.append("G7 public reproduction report changed")
    if repro.get("historical_reference_sha256") != expected_hash:
        errors.append("G7 reproduction historical-reference hash changed")
    if repro.get("reproduced_output_sha256") != expected_hash:
        errors.append("G7 reproduction output hash changed")

poc_path = ROOT / "results/reproduction/runtime_poc_seed4300_report.json"
if poc_path.exists():
    poc = json.loads(poc_path.read_text(encoding="utf-8"))
    if poc.get("status") != "PASS_WITH_BOUNDARY" or poc.get("hardware") != "CPU only":
        errors.append("runtime PoC scope/status changed")
    ser = poc.get("serialized_bytes_including_manifest", {})
    if ser.get("compact", 10**18) >= ser.get("large", -1):
        errors.append("runtime PoC compact artifact is no longer smaller")
    inf = poc.get("cpu_inference_batch128_ms", {})
    if inf.get("compact_mean", 10**18) >= inf.get("large_mean", -1):
        errors.append("runtime PoC CPU inference relation changed")
    interpretation = poc.get("interpretation", {})
    if "demonstrated" not in str(interpretation.get("host_ram", "")).lower():
        errors.append("runtime PoC host-RAM boundary missing")
    if "not established" not in str(interpretation.get("generalization", "")).lower():
        errors.append("runtime PoC generalization boundary missing")

vit_path = ROOT / "results/replication/vit_compositional/confirm_summary.json"
if vit_path.exists():
    rep = json.loads(vit_path.read_text(encoding="utf-8"))
    primary = rep.get("primary", {})
    secondary = rep.get("secondary", {})
    if rep.get("n") != 8 or primary.get("pass") is not True:
        errors.append("ViT compositional replication status changed")
    if primary.get("composed_lower_count") != 8:
        errors.append("ViT compositional replication must preserve 8/8 composed-lower result")
    ci = primary.get("paired_bootstrap95_ratio", [999, 999])
    if len(ci) != 2 or ci[1] >= 1.0:
        errors.append("ViT ratio CI must remain below 1")
    if secondary.get("mean_composed_test_utility", 0.0) < 0.95:
        errors.append("ViT selected composed test utility changed")

mlp_path = ROOT / "results/core_discovery_digits/confirm_summary.json"
if mlp_path.exists():
    rep = json.loads(mlp_path.read_text(encoding="utf-8"))
    primary = rep.get("primary", {})
    secondary = rep.get("secondary_confirmatory", {})
    mech = rep.get("mechanistic_secondary_2048_params", {})
    expected_lock = "f0d16d813918f2a419a8ba1dfd0bf0efe663a0b7f8105288a84dc84f18530f5b"
    if rep.get("status") != "PASS" or rep.get("original_protocol_lock_sha256") != expected_lock:
        errors.append("residual-MLP protocol/status changed")
    if primary.get("n") != 8 or primary.get("wins_composed_lower_budget") != 8:
        errors.append("residual-MLP must preserve n=8 and 8/8 composed-lower")
    ci = primary.get("bootstrap95", [999, 999])
    if len(ci) != 2 or ci[1] >= 0:
        errors.append("residual-MLP primary CI must remain below zero")
    if primary.get("mean_composed_budget", 10**18) >= primary.get("mean_componentwise_budget", -1):
        errors.append("residual-MLP mean budget relation changed")
    util_ci = secondary.get("bootstrap95", [-999, -999])
    if len(util_ci) != 2 or util_ci[0] <= secondary.get("noninferiority_margin", -0.02):
        errors.append("residual-MLP test-utility noninferiority changed")
    if not (
        mech.get("single_composed_mean_nmse", 10**18)
        < mech.get("joint_factorized_span_objective_mean_nmse", 10**18)
        < mech.get("local_componentwise_mean_nmse", -1)
    ):
        errors.append("residual-MLP mechanistic-control ordering changed")

correction_path = ROOT / "results/phase2/precision_composition/CORRECTION_STATUS.json"
if correction_path.exists():
    phases = json.loads(correction_path.read_text(encoding="utf-8")).get("phases", {})
    if phases.get("2E", {}).get("status") != "INVALIDATED_IMPLEMENTATION_BUG":
        errors.append("Phase 2E must remain INVALIDATED_IMPLEMENTATION_BUG")
    if phases.get("2E", {}).get("scientific_use") != "DO_NOT_USE_FOR_INFERENCE":
        errors.append("Phase 2E must remain excluded from inference")
    if phases.get("2I", {}).get("status") != "CAUSAL_CLAIM_RETRACTED":
        errors.append("Phase 2I causal claim must remain retracted")
    if phases.get("2O", {}).get("status") != "VALID_UNCERTAIN":
        errors.append("Phase 2O must remain VALID_UNCERTAIN")
    ci = phases.get("2O", {}).get("mean_difference_bootstrap95", [999, 999])
    if len(ci) != 2 or not (ci[0] < 0 < ci[1]):
        errors.append("Phase 2O interval must continue to cross zero")
    if phases.get("2O", {}).get("one_sided_exact_sign_test_p", 0.0) <= 0.05:
        errors.append("Phase 2O unexpectedly implies a significant advantage")

for item in (
    "README.md",
    "STATUS.md",
    "docs/CLAIMS_AND_EVIDENCE.md",
    "docs/phase2/README.md",
    "docs/NEGATIVE_RESULTS.md",
):
    p = ROOT / item
    if p.exists() and "INVALIDATED_IMPLEMENTATION_BUG" not in p.read_text(encoding="utf-8"):
        errors.append(f"Phase 2E invalidation marker missing: {item}")

# Structural hardening: the old version-number archaeology must stay out of the
# active tree once Issue #16 is applied.
for forbidden in (
    "docs/history",
    "docs/phases",
    "results/history",
    "results/phaseA_v11",
    "results/phaseB_v11",
    "results/v12",
    "results/v25",
    "environment/history",
    "scripts/phases/v11",
    "scripts/phases/v23",
):
    if (ROOT / forbidden).exists():
        errors.append(f"legacy path leaked back into active surface: {forbidden}")

if errors:
    print("AUDIT FAIL")
    for e in errors:
        print(" -", e)
    raise SystemExit(1)

print("AUDIT PASS")
print("python files:", sum(1 for _ in ROOT.rglob("*.py")))
print("markdown files:", sum(1 for _ in ROOT.rglob("*.md")))
print("csv files:", sum(1 for _ in ROOT.rglob("*.csv")))
print("json files:", sum(1 for _ in ROOT.rglob("*.json")))
