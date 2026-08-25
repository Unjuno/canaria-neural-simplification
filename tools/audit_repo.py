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

required = [
    ROOT / "README.md",
    ROOT / "STATUS.md",
    ROOT / "CHANGELOG.md",
    ROOT / "LICENSE",
    ROOT / "CITATION.cff",
    ROOT / "CONTRIBUTING.md",
    ROOT / "docs/README.md",
    ROOT / "docs/PUBLIC_SNAPSHOT.md",
    ROOT / "docs/HISTORICAL_INDEX.md",
    ROOT / "docs/CORE_DISCOVERY.md",
    ROOT / "docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md",
    ROOT / "docs/CORE_DISCOVERY_REPLICATION_DIGITS.md",
    ROOT / "docs/CLAIMS_AND_EVIDENCE.md",
    ROOT / "docs/PUBLICATION_NOTES.md",
    ROOT / "docs/TERMINOLOGY.md",
    ROOT / "docs/FAQ.md",
    ROOT / "docs/TRAINING_TIME_CONSOLIDATION.md",
    ROOT / "docs/LATE_STAGE_FINDINGS.md",
    ROOT / "docs/NEGATIVE_RESULTS.md",
    ROOT / "docs/APPLICATIONS.md",
    ROOT / "docs/RUNTIME_POC.md",
    ROOT / "docs/OPEN_QUESTIONS.md",
    ROOT / "docs/REPRODUCIBILITY.md",
    ROOT / "docs/ROADMAP.md",
    ROOT / "docs/RELEASE_CHECKLIST.md",
    ROOT / "results/phaseA_v11/stage3_confirmatory_summary.json",
    ROOT / "results/training_time/summary.json",
    ROOT / "results/training_time/protocol_manifest.json",
    ROOT / "results/training_time/late_stage_summary.json",
    ROOT / "results/training_time/ARTIFACT_INVENTORY.md",
    ROOT / "results/reproduction/README.md",
    ROOT / "scripts/reproduce/g7_confirmatory/run_seed.py",
    ROOT / "scripts/reproduce/g7_confirmatory/README.md",
    ROOT / "scripts/reproduce/g7_confirmatory/requirements.txt",
    ROOT / "scripts/reproduce/g7_confirmatory/runtime_poc.py",
    ROOT / "results/reproduction/g7_seed4300_report.json",
    ROOT / "results/reproduction/runtime_poc_seed4300_report.json",
    ROOT / ".github/workflows/reproduce-g7.yml",
    ROOT / ".github/workflows/runtime-poc.yml",
    ROOT / "scripts/replication/vit_compositional.py",
    ROOT / "results/replication/vit_compositional/PROTOCOL_LOCK.json",
    ROOT / "results/replication/vit_compositional/confirm_summary.json",
    ROOT / "results/replication/vit_compositional/seed_table.csv",
    ROOT / "scripts/reproduce/core_discovery_digits/run_confirmatory.py",
    ROOT / "results/core_discovery_digits/PROTOCOL_LOCK.json",
    ROOT / "results/core_discovery_digits/confirm_summary.json",
]
for p in required:
    if not p.exists():
        errors.append(f"missing required file: {rel(p)}")

public_markdown = [p for p in required if p.suffix == ".md"]
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
            errors.append(f"public markdown link escapes repository: {rel(p)} -> {raw_target}")
            continue
        if not candidate.exists():
            errors.append(f"broken public markdown link: {rel(p)} -> {raw_target}")

reproduce_root = ROOT / "scripts/reproduce"
if reproduce_root.exists():
    for p in reproduce_root.rglob("*.py"):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as exc:
            errors.append(f"portable runner read error: {rel(p)}: {exc}")
            continue
        if "/mnt/data" in text:
            errors.append(f"private /mnt/data dependency reintroduced in portable runner: {rel(p)}")

late_path = ROOT / "results/training_time/late_stage_summary.json"
if late_path.exists():
    try:
        late = json.loads(late_path.read_text(encoding="utf-8"))
        experiments = late.get("experiments", {})
        for name in ("G18", "G19", "G20d", "G20e", "G21", "G22", "G23", "G24", "G25", "G26"):
            if name not in experiments:
                errors.append(f"late-stage manifest missing experiment: {name}")
        if experiments.get("G21", {}).get("status") != "FAIL":
            errors.append("late-stage manifest must preserve G21 as FAIL")
    except Exception as exc:
        errors.append(f"late-stage semantic audit: {exc}")

repro_path = ROOT / "results/reproduction/g7_seed4300_report.json"
if repro_path.exists():
    try:
        repro = json.loads(repro_path.read_text(encoding="utf-8"))
        expected_hash = "68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028"
        if repro.get("status") != "PASS":
            errors.append("G7 public reproduction report must remain PASS")
        if repro.get("exact_json_equal") is not True:
            errors.append("G7 public reproduction report must preserve exact_json_equal=true")
        if repro.get("historical_reference_sha256") != expected_hash:
            errors.append("G7 reproduction historical-reference hash changed")
        if repro.get("reproduced_output_sha256") != expected_hash:
            errors.append("G7 reproduction output hash changed")
    except Exception as exc:
        errors.append(f"G7 reproduction semantic audit: {exc}")

poc_path = ROOT / "results/reproduction/runtime_poc_seed4300_report.json"
if poc_path.exists():
    try:
        poc = json.loads(poc_path.read_text(encoding="utf-8"))
        if poc.get("status") != "PASS_WITH_BOUNDARY":
            errors.append("runtime PoC must remain PASS_WITH_BOUNDARY")
        if poc.get("hardware") != "CPU only":
            errors.append("runtime PoC hardware scope changed unexpectedly")
        ser = poc.get("serialized_bytes_including_manifest", {})
        if ser.get("compact", 10**18) >= ser.get("large", -1):
            errors.append("runtime PoC must preserve smaller compact serialized artifact")
        inf = poc.get("cpu_inference_batch128_ms", {})
        if inf.get("compact_mean", 10**18) >= inf.get("large_mean", -1):
            errors.append("runtime PoC must preserve measured compact CPU inference advantage")
        interpretation = poc.get("interpretation", {})
        host_ram = str(interpretation.get("host_ram", "")).lower()
        generalization = str(interpretation.get("generalization", "")).lower()
        if "not" not in host_ram or "demonstrated" not in host_ram:
            errors.append("runtime PoC must preserve host-RAM non-claim")
        if "not established" not in generalization:
            errors.append("runtime PoC must preserve generalization boundary")
    except Exception as exc:
        errors.append(f"runtime PoC semantic audit: {exc}")

vit_path = ROOT / "results/replication/vit_compositional/confirm_summary.json"
if vit_path.exists():
    try:
        rep = json.loads(vit_path.read_text(encoding="utf-8"))
        primary = rep.get("primary", {})
        secondary = rep.get("secondary", {})
        if rep.get("n") != 8:
            errors.append("ViT compositional replication must preserve n=8")
        if primary.get("pass") is not True:
            errors.append("ViT compositional replication must remain PASS")
        if primary.get("composed_lower_count") != 8:
            errors.append("ViT compositional replication must preserve 8/8 composed-lower result")
        ci = primary.get("paired_bootstrap95_ratio", [999, 999])
        if len(ci) != 2 or ci[1] >= 1.0:
            errors.append("ViT compositional replication bootstrap CI must remain below 1")
        if secondary.get("mean_composed_test_utility", 0.0) < 0.95:
            errors.append("ViT compositional replication must preserve mean composed test utility >=0.95")
        if secondary.get("mean_composed_compiler_updates", 10**9) >= secondary.get("mean_componentwise_compiler_updates", -1):
            errors.append("ViT replication must preserve lower composed compiler-update count")
    except Exception as exc:
        errors.append(f"ViT compositional replication semantic audit: {exc}")

mlp_path = ROOT / "results/core_discovery_digits/confirm_summary.json"
if mlp_path.exists():
    try:
        rep = json.loads(mlp_path.read_text(encoding="utf-8"))
        primary = rep.get("primary", {})
        secondary = rep.get("secondary_confirmatory", {})
        mech = rep.get("mechanistic_secondary_2048_params", {})
        expected_lock = "f0d16d813918f2a419a8ba1dfd0bf0efe663a0b7f8105288a84dc84f18530f5b"
        if rep.get("status") != "PASS":
            errors.append("residual-MLP compositional replication must remain PASS")
        if rep.get("original_protocol_lock_sha256") != expected_lock:
            errors.append("residual-MLP original protocol lock hash changed")
        if primary.get("n") != 8:
            errors.append("residual-MLP compositional replication must preserve n=8")
        if primary.get("wins_composed_lower_budget") != 8:
            errors.append("residual-MLP replication must preserve 8/8 composed-lower result")
        ci = primary.get("bootstrap95", [999, 999])
        if len(ci) != 2 or ci[1] >= 0:
            errors.append("residual-MLP primary log-budget CI must remain below zero")
        if primary.get("mean_composed_budget", 10**18) >= primary.get("mean_componentwise_budget", -1):
            errors.append("residual-MLP replication must preserve lower mean composed budget")
        util_ci = secondary.get("bootstrap95", [-999, -999])
        if len(util_ci) != 2 or util_ci[0] <= secondary.get("noninferiority_margin", -0.02):
            errors.append("residual-MLP test-utility noninferiority result changed")
        local = mech.get("local_componentwise_mean_nmse", -1)
        joint = mech.get("joint_factorized_span_objective_mean_nmse", 10**18)
        composed = mech.get("single_composed_mean_nmse", 10**18)
        if not (joint < local):
            errors.append("residual-MLP joint span-objective control must remain better than local component-wise fit")
        if not (composed < joint):
            errors.append("residual-MLP single composed control must remain lower NMSE than joint factorized control")
    except Exception as exc:
        errors.append(f"residual-MLP compositional replication semantic audit: {exc}")

for p in (ROOT / "README.md", ROOT / "STATUS.md"):
    if p.exists():
        text = p.read_text(encoding="utf-8")
        if "public-snapshot" not in text and "public snapshot" not in text:
            errors.append(f"snapshot-state marker missing: {rel(p)}")

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
