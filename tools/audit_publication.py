"""Check publication integrity; optional strict readiness gate remains separate."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote,urlsplit
ROOT=Path(__file__).resolve().parents[1]
IGNORED={'.git','.venv','venv','__pycache__','.pytest_cache','build','dist','outputs'}


def safe_path(root:Path,name:str)->Path:
    p=(root/name).resolve()
    if not p.is_relative_to(root.resolve()):raise ValueError('path escapes root: '+name)
    return p


def check_preservation(root:Path,manifest:dict)->list[str]:
    errors=[];seen=set()
    for rec in manifest['files']:
        name=rec['path']
        if name in seen:errors.append('duplicate preservation path: '+name)
        seen.add(name)
        try:
            p=safe_path(root,name);raw=p.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=rec['sha256']:errors.append('changed baseline bytes: '+name)
            blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if blob!=rec['git_blob_sha1']:errors.append('changed Git blob: '+name)
        except (OSError,ValueError) as e:errors.append(str(e))
    return errors


def local_link_errors(root:Path)->list[str]:
    errors=[]
    for p in root.rglob('*.md'):
        parts=set(p.relative_to(root).parts)
        if parts & IGNORED or 'archives' in parts:continue
        text=re.sub(r'```.*?```','',p.read_text(),flags=re.S)
        for raw in re.findall(r'\[[^\]]*\]\(([^)]+)\)',text):
            url=raw.strip().split()[0].strip('<>')
            parsed=urlsplit(url)
            if parsed.scheme or not parsed.path:continue
            try:
                rel=str((p.parent/unquote(parsed.path)).resolve().relative_to(root.resolve()))
                if not safe_path(root,rel).exists():errors.append(f'broken local link: {p.relative_to(root)} -> {raw}')
            except ValueError:errors.append(f'link escapes root: {p.relative_to(root)} -> {raw}')
    return errors


def audit(root:Path,require_reproduction:bool=False)->dict:
    errors=[]
    policy=json.loads((root/'publication/CLAIM_POLICY.json').read_text())
    manifest=json.loads((root/'publication/PRESERVATION_MANIFEST.json').read_text())
    catalog=json.loads((root/'publication/RESEARCH_CATALOG.json').read_text())
    errors+=check_preservation(root,manifest);errors+=local_link_errors(root)
    if manifest['baseline_commit']!='41872aca00ee5750556c93e114f705bce2e9c611':errors.append('baseline changed')
    if policy['headline_experiments']!=['CORE_DIGITS_COMPONENT_VS_COMPOSED']:errors.append('unreviewed headline promotion')
    for key in ['research_promoted_to_baseline','independent_external_reproduction_claimed','peer_review_claimed','new_hardware_claims','announcement_automatically_posted']:
        if policy.get(key) is not False:errors.append('unexpected policy claim: '+key)
    for rec in catalog['entries']:
        if not re.fullmatch('[0-9a-f]{40}',rec['commit']):errors.append('unpinned research source: '+rec['id'])
        for name in rec['paths']:
            try:safe_path(root,name)
            except ValueError as e:errors.append(str(e))
    for old in ['REVIEW_HANDOFF.md','docs/PUBLIC_SNAPSHOT.md','docs/RELEASE_CHECKLIST.md','docs/history','docs/phases','results/v25']:
        if (root/old).exists():errors.append('historical material still in active path: '+old)
    for name in ['README.md','README.ja.md','STATUS.md','docs/ANNOUNCEMENT_READINESS.md']:
        text=(root/name).read_text()
        if 'REPRODUCTION_DISCREPANCY.md' not in text:errors.append('missing reproduction disclosure: '+name)
    if policy.get('mode')!='PRE_ANNOUNCEMENT_REPRODUCTION_BLOCKED' or policy.get('exact_cohort_reproduction_established') is not False or 87 not in policy.get('blockers',[]):
        errors.append('known blocker silently removed')
    if require_reproduction:
        report=root/policy['reproduction_report']
        if not report.exists():errors.append('strict readiness: current full-cohort report missing')
        else:
            d=json.loads(report.read_text())
            if d.get('status')!='PASS':errors.append('strict readiness: full-cohort numerical reproduction is not PASS')
            if d.get('evidence_class')!='reproduction_of_existing_confirmatory_cohort':errors.append('report evidence class changed')
            if d.get('seeds')!=list(range(1200,1208)) or len(d.get('rows',[]))!=8:errors.append('full cohort not present')
            if not d.get('input_sha256'):errors.append('report lacks scientific input hashes')
            for name,sha in d.get('input_sha256',{}).items():
                if hashlib.sha256(safe_path(root,name).read_bytes()).hexdigest()!=sha:errors.append('report input drift: '+name)
        errors.append('strict readiness: Issue87 remains explicitly unresolved; review required')
    return {'integrity_status':'FAIL' if errors else 'PASS','announcement_status':'BLOCKED_REPRODUCTION_ISSUE_87','protected_file_count':len(manifest['files']),'research_entry_count':len(catalog['entries']),'errors':errors,'scope':'File/link/policy preservation audit, not independent scientific review or exact reproduction.'}


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--require-reproduction',action='store_true');ap.add_argument('--out',type=Path)
    a=ap.parse_args();d=audit(ROOT,a.require_reproduction)
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(d,indent=2));return 0 if d['integrity_status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
