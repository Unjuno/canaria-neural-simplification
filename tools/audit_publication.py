"""Check publication-candidate integrity while preserving the separate historical reproduction gate."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', 'build', 'dist', 'outputs'}


def safe_path(root: Path, name: str) -> Path:
    p = (root / name).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError('path escapes root: ' + name)
    return p


def check_preservation(root: Path, manifest: dict) -> list[str]:
    errors = []
    seen = set()
    for rec in manifest['files']:
        name = rec['path']
        if name in seen:
            errors.append('duplicate preservation path: ' + name)
        seen.add(name)
        try:
            p = safe_path(root, name)
            raw = p.read_bytes()
            if hashlib.sha256(raw).hexdigest() != rec['sha256']:
                errors.append('changed baseline bytes: ' + name)
            blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
            if blob != rec['git_blob_sha1']:
                errors.append('changed Git blob: ' + name)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    return errors


def local_link_errors(root: Path) -> list[str]:
    errors = []
    for p in root.rglob('*.md'):
        parts = set(p.relative_to(root).parts)
        if parts & IGNORED or 'archives' in parts:
            continue
        text = re.sub(r'```.*?```', '', p.read_text(), flags=re.S)
        for raw in re.findall(r'\[[^\]]*\]\(([^)]+)\)', text):
            url = raw.strip().split()[0].strip('<>')
            parsed = urlsplit(url)
            if parsed.scheme or not parsed.path:
                continue
            try:
                rel = str((p.parent / unquote(parsed.path)).resolve().relative_to(root.resolve()))
                if not safe_path(root, rel).exists():
                    errors.append(f'broken local link: {p.relative_to(root)} -> {raw}')
            except ValueError:
                errors.append(f'link escapes root: {p.relative_to(root)} -> {raw}')
    return errors


def audit(root: Path, require_reproduction: bool = False) -> dict:
    errors = []
    policy = json.loads((root / 'publication/CLAIM_POLICY.json').read_text())
    manifest = json.loads((root / 'publication/PRESERVATION_MANIFEST.json').read_text())
    catalog = json.loads((root / 'publication/RESEARCH_CATALOG.json').read_text())

    errors += check_preservation(root, manifest)
    errors += local_link_errors(root)

    if manifest['baseline_commit'] != '41872aca00ee5750556c93e114f705bce2e9c611':
        errors.append('preservation baseline changed')
    if policy.get('schema_version') != 2 or policy.get('mode') != 'REVIEWED_PUBLICATION_CANDIDATE':
        errors.append('unexpected active publication policy')
    if policy.get('review_base_main') != 'c7fd4fb701064c9aeef5cc8109231d884e5258aa':
        errors.append('review base main changed')
    if policy.get('headline_experiments') != ['R87R2_FRESH_SQRT_PROFILE_CONFIRMATION']:
        errors.append('unreviewed headline selection')
    if policy.get('bounded_external_validity_support') != ['PHASE4_CALIFORNIA_CONFIRMATORY']:
        errors.append('external-validity support selection changed')

    for key in [
        'independent_external_reproduction_claimed',
        'peer_review_claimed',
        'new_hardware_claims',
        'architecture_universality_claimed',
        'runtime_speedup_claimed',
        'ram_vram_energy_claimed',
        'llm_claimed',
        'announcement_automatically_posted',
        'release_tag_created',
        'merged_to_main_by_this_candidate',
    ]:
        if policy.get(key) is not False:
            errors.append('unexpected policy claim: ' + key)

    hist = policy.get('historical_archive', {})
    if hist.get('id') != 'HISTORICAL_CORE_DIGITS_1200_1207':
        errors.append('historical archive identity changed')
    if hist.get('exact_reproduction_established') is not False:
        errors.append('historical archive falsely marked reproduced')
    if hist.get('issue') != 87 or hist.get('issue_remains_open') is not True:
        errors.append('Issue 87 silently removed')
    if hist.get('blocks_current_r87r2_headline') is not False:
        errors.append('reviewed R87R2 headline incorrectly blocked by historical debt')
    if hist.get('r87r2_is_historical_recovery') is not False:
        errors.append('R87R2 falsely relabeled historical recovery')

    headline_repro = policy.get('headline_reproduction', {})
    expected_repro = {
        'workflow': '.github/workflows/publication-headline-reproduction.yml',
        'quickstart': 'QUICKSTART.md',
        'requirements': 'scripts/reproduction_diagnostics/requirements-r87r2-py313.txt',
        'python': '3.13.5',
        'torch_cpu': '2.10.0',
        'cohort': '871200-871215',
        'comparison': 'exact scientific outcome objects and aggregate decision versus reviewed vendored primary evidence',
        'evidence_class': 'TECHNICAL_REPRODUCTION_OF_FIXED_FRESH_COHORT',
        'adds_independent_scientific_seeds': False,
    }
    if headline_repro != expected_repro:
        errors.append('headline clean-reproduction policy changed')
    for name in (expected_repro['workflow'], expected_repro['quickstart'], expected_repro['requirements']):
        if not safe_path(root, name).exists():
            errors.append('missing headline reproduction surface: ' + name)

    expected_requirements = [
        'numpy==2.3.5',
        'scikit-learn==1.8.0',
        'scipy==1.17.0',
        'joblib==1.5.3',
        'threadpoolctl==3.6.0',
        'sympy==1.14.0',
        'networkx==3.6.1',
        'filelock==3.29.0',
        'fsspec==2026.4.0',
        'jinja2==3.1.6',
        'typing_extensions==4.16.0',
    ]
    req_path = safe_path(root, expected_repro['requirements'])
    if req_path.exists() and req_path.read_text().splitlines() != expected_requirements:
        errors.append('R87R2 pinned requirements changed')

    workflow_path = safe_path(root, expected_repro['workflow'])
    if workflow_path.exists():
        workflow = workflow_path.read_text()
        for token in [
            "python-version: '3.13.5'",
            'torch==2.10.0',
            'r87r2_confirm.py --mode suite --workers 2',
            'rows_exact_to_reviewed_primary',
            'decision_exact_to_reviewed_primary',
            "'new_independent_scientific_seeds': 0",
            "'historical_archive_recovered': False",
        ]:
            if token not in workflow:
                errors.append('headline reproduction workflow missing locked token: ' + token)

    for rec in catalog['entries']:
        if not re.fullmatch('[0-9a-f]{40}', rec['commit']):
            errors.append('unpinned research source: ' + rec['id'])
        for name in rec['paths']:
            try:
                safe_path(root, name)
            except ValueError as exc:
                errors.append(str(exc))

    for old in ['REVIEW_HANDOFF.md', 'docs/PUBLIC_SNAPSHOT.md', 'docs/RELEASE_CHECKLIST.md', 'docs/history', 'docs/phases', 'results/v25']:
        if (root / old).exists():
            errors.append('historical material still in active path: ' + old)

    surface = ['README.md', 'README.ja.md', 'STATUS.md', 'QUICKSTART.md', 'docs/CLAIMS_AND_EVIDENCE.md', 'docs/ANNOUNCEMENT_READINESS.md']
    for name in surface:
        text = (root / name).read_text()
        if 'REPRODUCTION_DISCREPANCY.md' not in text:
            errors.append('missing historical reproduction disclosure: ' + name)
        if 'R87R2' not in text:
            errors.append('missing current reviewed baseline disclosure: ' + name)
        if 'Phase4' not in text and 'PHASE4' not in text:
            errors.append('missing bounded Phase4 disclosure: ' + name)

    citation = (root / 'CITATION.cff').read_text()
    if 'rolling research preview is not a peer-reviewed paper' not in citation:
        errors.append('CITATION.cff missing peer-review boundary')
    if 'No universal compression or general hardware-resource advantage is claimed.' not in citation:
        errors.append('CITATION.cff missing scope boundary')

    for required in [policy.get('review_ledger'), policy.get('candidate_evidence_manifest'), policy.get('candidate_evidence_gate')]:
        if not required:
            errors.append('missing candidate evidence reference in policy')
        else:
            try:
                if not safe_path(root, required).exists():
                    errors.append('missing candidate evidence reference: ' + required)
            except ValueError as exc:
                errors.append(str(exc))

    # Backward-compatible strict historical reproduction gate. This is retained
    # for Issue #87 and is intentionally NOT the gate for the new R87R2 headline.
    if require_reproduction:
        report_name = hist.get('reproduction_report')
        if not report_name:
            errors.append('strict historical reproduction: report path missing')
        else:
            report = safe_path(root, report_name)
            if not report.exists():
                errors.append('strict historical reproduction: current full-cohort report missing')
            else:
                d = json.loads(report.read_text())
                if d.get('status') != 'PASS':
                    errors.append('strict historical reproduction: full-cohort numerical reproduction is not PASS')
                if d.get('evidence_class') != 'reproduction_of_existing_confirmatory_cohort':
                    errors.append('strict historical reproduction: report evidence class changed')
                if d.get('seeds') != list(range(1200, 1208)) or len(d.get('rows', [])) != 8:
                    errors.append('strict historical reproduction: full cohort not present')
                if not d.get('input_sha256'):
                    errors.append('strict historical reproduction: report lacks scientific input hashes')
                for name, sha in d.get('input_sha256', {}).items():
                    if hashlib.sha256(safe_path(root, name).read_bytes()).hexdigest() != sha:
                        errors.append('strict historical reproduction: report input drift: ' + name)
        errors.append('strict historical reproduction: Issue 87 remains explicitly unresolved')

    ok = not errors
    return {
        'integrity_status': 'PASS' if ok else 'FAIL',
        'announcement_status': 'CANDIDATE_INTEGRITY_PASS_FINAL_REVIEW_REQUIRED' if ok else 'CANDIDATE_INTEGRITY_FAIL',
        'historical_reproduction_status': 'UNRESOLVED_ISSUE_87',
        'historical_reproduction_blocks_current_r87r2_headline': False,
        'headline_clean_reproduction_required': True,
        'protected_file_count': len(manifest['files']),
        'research_entry_count': len(catalog['entries']),
        'errors': errors,
        'scope': 'File/link/policy preservation audit. R87R2 and Phase4 raw-row scientific recalculation is performed by tools/audit_publication_candidate.py; clean-checkout R87R2 retraining is performed by publication-headline-reproduction CI; strict historical 1200-1207 reproduction remains available via --require-reproduction.'
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--require-reproduction', action='store_true', help='Run the separate strict historical 1200-1207 Issue #87 gate; expected to fail while Issue #87 is unresolved.')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    d = audit(ROOT, args.require_reproduction)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(d, indent=2) + '\n')
    print(json.dumps(d, indent=2))
    return 0 if d['integrity_status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
