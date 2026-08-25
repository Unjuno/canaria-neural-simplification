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
    ROOT / "LICENSE",
    ROOT / "CITATION.cff",
    ROOT / "CONTRIBUTING.md",
    ROOT / "docs/PUBLIC_SNAPSHOT.md",
    ROOT / "docs/HISTORICAL_INDEX.md",
    ROOT / "docs/CORE_DISCOVERY.md",
    ROOT / "docs/CLAIMS_AND_EVIDENCE.md",
    ROOT / "docs/TRAINING_TIME_CONSOLIDATION.md",
    ROOT / "docs/LATE_STAGE_FINDINGS.md",
    ROOT / "docs/NEGATIVE_RESULTS.md",
    ROOT / "docs/APPLICATIONS.md",
    ROOT / "docs/OPEN_QUESTIONS.md",
    ROOT / "docs/REPRODUCIBILITY.md",
    ROOT / "docs/ROADMAP.md",
    ROOT / "results/phaseA_v11/stage3_confirmatory_summary.json",
    ROOT / "results/training_time/summary.json",
    ROOT / "results/training_time/protocol_manifest.json",
    ROOT / "results/training_time/late_stage_summary.json",
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
