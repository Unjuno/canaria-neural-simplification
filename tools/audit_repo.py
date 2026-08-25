from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

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
    ROOT / "docs/PUBLIC_SNAPSHOT.md",
    ROOT / "docs/HISTORICAL_INDEX.md",
    ROOT / "docs/CORE_DISCOVERY.md",
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
]
for p in required:
    if not p.exists():
        errors.append(f"missing required file: {rel(p)}")

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
