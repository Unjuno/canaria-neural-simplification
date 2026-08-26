from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []

required = [
    ROOT / "docs/ANNOUNCEMENT_READINESS.md",
    ROOT / "scripts/reproduce/core_discovery_digits/README.md",
    ROOT / "scripts/reproduce/core_discovery_digits/requirements-pinned-py311.txt",
    ROOT / "scripts/reproduce/core_discovery_digits/verify_confirmatory.py",
    ROOT / "archives/README.md",
    ROOT / "archives/releases/v0.2.0/RELEASE_CHECKLIST.md",
]
for path in required:
    if not path.exists():
        errors.append(f"missing readiness/provenance path: {path.relative_to(ROOT)}")

surfaces = {
    "README.md": ROOT / "README.md",
    "STATUS.md": ROOT / "STATUS.md",
    "docs/ANNOUNCEMENT_READINESS.md": ROOT / "docs/ANNOUNCEMENT_READINESS.md",
}
for name, path in surfaces.items():
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8").lower()
    if "announcement" not in text:
        errors.append(f"announcement-state wording missing: {name}")
    if name != "README.md" and "issue #13" not in text:
        errors.append(f"Issue #13 readiness gate missing: {name}")

stale_forbidden = {
    "README.md": ["current mode: reviewed public baseline"],
    "STATUS.md": ["current mode: reviewed public baseline", "publication sequence is complete"],
    "docs/README.md": ["current post-snapshot public baseline"],
}
for rel, phrases in stale_forbidden.items():
    p = ROOT / rel
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8").lower()
    for phrase in phrases:
        if phrase in text:
            errors.append(f"stale announcement-ready wording in {rel}: {phrase}")

req = ROOT / "scripts/reproduce/core_discovery_digits/requirements-pinned-py311.txt"
if req.exists():
    lines = {
        line.strip()
        for line in req.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    expected = {"numpy==2.4.6", "scikit-learn==1.9.0", "torch==2.13.0"}
    if lines != expected:
        errors.append("pinned core-digits requirements changed without an explicit readiness update")

historical_checklist = ROOT / "archives/releases/v0.2.0/RELEASE_CHECKLIST.md"
if historical_checklist.exists():
    text = historical_checklist.read_text(encoding="utf-8").lower()
    if "historical" not in text or "not the current announcement" not in text:
        errors.append("archived release checklist is not clearly separated from current readiness")

# Historical gate files must not reappear as active root/docs gates.
for stale_path in (ROOT / "REVIEW_HANDOFF.md", ROOT / "docs/RELEASE_CHECKLIST.md", ROOT / "docs/PUBLIC_SNAPSHOT.md"):
    if stale_path.exists():
        errors.append(f"historical gate file leaked back into active surface: {stale_path.relative_to(ROOT)}")

report = ROOT / "results/reproduction/core_discovery_digits_pinned_env_report.json"
if report.exists():
    try:
        data = json.loads(report.read_text(encoding="utf-8"))
        if data.get("status") != "PASS":
            errors.append("checked-in pinned core-digits reproduction report is not PASS")
        if data.get("evidence_class") != "reproduction_of_existing_confirmatory_cohort":
            errors.append("core-digits reproduction evidence class changed unexpectedly")
    except Exception as exc:
        errors.append(f"core-digits reproduction report parse failure: {exc}")

if errors:
    print("READINESS AUDIT FAIL")
    for error in errors:
        print(" -", error)
    raise SystemExit(1)

print("READINESS AUDIT PASS")
