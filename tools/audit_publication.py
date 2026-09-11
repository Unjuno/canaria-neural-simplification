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

    surface = ['README.md', 'README.ja.md', 'STATUS.md', 'docs/CLAIMS_AND_EVIDENCE.md', 'docs/ANNOUNCEMENT_READINESS.md']
    for name in surface:
        text = (root / name).read_text()
        if 'REPRODUCTION_DISCREPANCY.md' not in text:
            errors.append('missing historical reproduction disclosure: ' + name)
        if 'R87R2' not in text:
            errors.append('missing current reviewed baseline disclosure: ' + name)
        if 'Phase4' not in text and 'PHASE4' not in text:
            errors.append('missing bounded Phase4 disclosure: ' + name)

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
        'protected_file_count': len(manifest['files']),
        'research_entry_count': len(catalog['entries']),
        'errors': errors,
        'scope': 'File/link/policy preservation audit. R87R2 and Phase4 raw-row scientific recalculation is performed by tools/audit_publication_candidate.py; strict historical 1200-1207 reproduction remains available via --require-reproduction.'
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
