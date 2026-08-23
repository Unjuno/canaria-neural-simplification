from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors = []

for p in ROOT.rglob("*.py"):
    if ".git" in p.parts:
        continue
    try:
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
    except Exception as exc:
        errors.append(f"python syntax: {p.relative_to(ROOT)}: {exc}")

for p in ROOT.rglob("*.json"):
    try:
        json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"json: {p.relative_to(ROOT)}: {exc}")

for p in ROOT.rglob("*.csv"):
    try:
        with p.open(newline="", encoding="utf-8") as f:
            list(csv.reader(f))
    except Exception as exc:
        errors.append(f"csv: {p.relative_to(ROOT)}: {exc}")

required = [
    ROOT / "README.md",
    ROOT / "LICENSE",
    ROOT / "docs/RESEARCH_SUMMARY.md",
    ROOT / "results/phaseA_v11/stage3_confirmatory_summary.json",
]
for p in required:
    if not p.exists():
        errors.append(f"missing required file: {p.relative_to(ROOT)}")

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
