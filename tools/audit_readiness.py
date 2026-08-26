from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
errors: list[str] = []

required = [
    ROOT / "docs/ANNOUNCEMENT_READINESS.md",
    ROOT / "scripts/reproduce/core_discovery_digits/README.md",
    ROOT / "scripts/reproduce/core_discovery_digits/requirements-pinned-py311.txt",
    ROOT / "scripts/reproduce/core_discovery_digits/verify_confirmatory.py",
]
for path in required:
    if not path.exists():
        errors.append(f"missing readiness file: {path.relative_to(ROOT)}")

surfaces = {
    "README.md": ROOT / "README.md",
    "STATUS.md": ROOT / "STATUS.md",
    "docs/ANNOUNCEMENT_READINESS.md": ROOT / "docs/ANNOUNCEMENT_READINESS.md",
}
for name, path in surfaces.items():
    if path.exists():
        text = path.read_text(encoding="utf-8").lower()
        if "announcement" not in text:
            errors.append(f"announcement-state wording missing: {name}")
        if "issue #13" not in text and name != "README.md":
            errors.append(f"Issue #13 readiness gate missing: {name}")

stale_forbidden = {
    "README.md": ["current mode: reviewed public baseline"],
    "STATUS.md": [
        "current mode: reviewed public baseline",
        "publication sequence is complete",
    ],
    "docs/README.md": ["current post-snapshot public baseline"],
}
for rel, phrases in stale_forbidden.items():
    path = ROOT / rel
    if not path.exists():
        continue
    text = path.read_text(encoding="utf-8").lower()
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
        errors.append(
            "pinned core-digits requirements changed; update readiness documentation and reproduction evidence explicitly"
        )

release_checklist = ROOT / "docs/RELEASE_CHECKLIST.md"
if release_checklist.exists():
    text = release_checklist.read_text(encoding="utf-8").lower()
    if "historical" not in text or "not the current announcement" not in text:
        errors.append("historical release checklist is not clearly separated from current readiness")

report = ROOT / "results/reproduction/core_discovery_digits_pinned_env_report.json"
if report.exists():
    import json

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
