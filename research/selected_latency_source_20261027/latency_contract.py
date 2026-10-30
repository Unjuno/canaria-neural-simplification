"""Versioned, local-only input/attempt contract. No numerical imports or file writes."""
from __future__ import annotations
import datetime,hashlib,json,os,pathlib,re,subprocess
import storage_manifest_contract as storage

SCHEMA='canaria.selected-latency-inputs.v1'
RELEASE_SCHEMA='canaria.selected-latency-release.v1'
CSV_SHA256='09f66e6debdee2cd2b5ae59e0d6abbb73fc2b0e0185d2e1957e9ebb51e23aa22'
ORIGIN_SELECTION_SHA256='d31543ed32bef5705f92f1a80974ad63bbca3c46e12d60b62d446c29f1e8bd1e'
REFERENCE_MANIFEST_SHA256='d7b132c953282fd0179b4da6e1e3226f60d5bad05128df5e81027ae3a50af761'
REFERENCE_SELECTION_SHA256='6cf4c87e8c7ca43f8760fffae8ea1155dfcccf47e61f778f2d4cc436d50d01f3'
ARMS=('teacher','C','J')
require=storage.require
exact_keys=storage.exact_keys
validate_digest=storage.validate_digest


def sha(path):
    digest=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def relative_path(value):
    require(type(value) is str and re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_-][A-Za-z0-9_.-]*)*',value) is not None,'invalid relative path')
    require(all(x not in ('.','..') for x in value.split('/')),'path traversal')
    return value


def contained(root,relative,directory=False):
    root=pathlib.Path(root).resolve(strict=True)
    require(root.is_dir(),'input root directory')
    relative_path(relative)
    target=root
    for part in relative.split('/'):
        target=target/part
        require(not target.is_symlink(),'symlinks forbidden')
    target=target.resolve(strict=True)
    require(target.is_relative_to(root),'input containment')
    require(target.is_dir() if directory else target.is_file(),'input type')
    return target


def descriptor(value,extra=()):
    exact_keys(value,('path','sha256',*extra),'file descriptor')
    relative_path(value['path']);validate_digest(value['sha256'])
    return value


def read_json(path,expected=None):
    raw=pathlib.Path(path).read_bytes()
    require(len(raw)<=4*1024**2,'JSON byte ceiling')
    if expected is not None:
        validate_digest(expected)
        require(hashlib.sha256(raw).hexdigest()==expected,'JSON SHA256 mismatch')
    return json.loads(raw,object_pairs_hook=storage.no_duplicate_fields,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def verify_file(root,value):
    path=contained(root,value['path'])
    require(sha(path)==value['sha256'],'input SHA256 mismatch')
    return path


def validate_inputs(manifest):
    exact_keys(manifest,('schema','selection','origin_selection','historical_reference_manifest','storage','csv','reference_selection','validation'),'latency inputs')
    require(manifest['schema']==SCHEMA,'input schema')
    for key in ('selection','origin_selection','historical_reference_manifest','csv','reference_selection'):descriptor(manifest[key])
    for key,expected in (('origin_selection',ORIGIN_SELECTION_SHA256),('historical_reference_manifest',REFERENCE_MANIFEST_SHA256),('reference_selection',REFERENCE_SELECTION_SHA256)):
        require(manifest[key]['sha256']==expected,'historical provenance identity')
    require(manifest['csv']['sha256']==CSV_SHA256,'frozen CSV identity')
    exact_keys(manifest['storage'],('directory','artifacts','rows'),'storage')
    relative_path(manifest['storage']['directory'])
    for key in ('artifacts','rows'):descriptor(manifest['storage'][key])
    rows=manifest['validation']
    require(type(rows) is list and len(rows)==16,'validation census')
    ids=[]
    for row in rows:
        exact_keys(row,('id','split','logits'),'validation row')
        require(type(row['id']) is int and 0<=row['id']<16,'validation id');ids.append(row['id'])
        descriptor(row['split']);exact_keys(row['logits'],ARMS,'logit census')
        for arm in ARMS:
            descriptor(row['logits'][arm],('key',))
            require(type(row['logits'][arm]['key']) is str,'logit key type')
    require(sorted(ids)==list(range(16)),'validation id uniqueness')
    return manifest


def load_inputs(manifest_path,expected_sha256,input_root):
    """Hash all required bytes before any archive/model loading; never select models."""
    manifest=validate_inputs(read_json(manifest_path,expected_sha256))
    selection_path=verify_file(input_root,manifest['selection'])
    selection=storage.read_manifest(selection_path,manifest['selection']['sha256'])
    require(selection['selection_origin_sha256']==ORIGIN_SELECTION_SHA256,'selection origin binding')
    origin=read_json(verify_file(input_root,manifest['origin_selection']))
    reference_manifest=read_json(verify_file(input_root,manifest['historical_reference_manifest']))
    reference_selection=read_json(verify_file(input_root,manifest['reference_selection']))
    # Historical path strings are lookup keys only: NEVER resolve/open those paths.
    origins={r['id']:r for r in origin};reference_choices={r['id']:r for r in reference_selection}
    require(sorted(origins)==list(range(16)) and len(origin)==16,'original choice census')
    require(sorted(reference_choices)==list(range(16)) and len(reference_selection)==16,'reference choice census')
    require(type(reference_manifest) is dict and len(reference_manifest)==47,'historical reference census')
    csv_path=verify_file(input_root,manifest['csv'])
    store=manifest['storage'];storage_root=contained(input_root,store['directory'],directory=True)
    artifacts=read_json(verify_file(input_root,store['artifacts']))
    rows=read_json(verify_file(input_root,store['rows']))
    require(type(artifacts) is list and len(artifacts)==96,'exact stored+deflated artifact census')
    artifact_map={}
    expected_names={f'seed_{i}_{a}_{label}.npz' for i in range(16) for a in ARMS for label in ('stored','deflated')}
    for item in artifacts:
        exact_keys(item,('file','bytes','sha256'),'artifact')
        require(item['file'] in expected_names and item['file'] not in artifact_map,'artifact identity')
        validate_digest(item['sha256']);require(type(item['bytes']) is int and item['bytes']>0,'artifact size')
        artifact_map[item['file']]=item
    require(type(rows) is list and len(rows)==16,'storage row census')
    require(all(type(r) is dict and type(r.get('id')) is int for r in rows),'storage row id')
    require(sorted(r['id'] for r in rows)==list(range(16)),'storage row uniqueness')
    row_map={r['id']:r for r in rows}
    validation={r['id']:r for r in manifest['validation']}
    resolved=[]
    for s in sorted(selection['models'],key=lambda r:r['id']):
        sid=s['id'];expected=row_map[sid];v=validation[sid];old=origins[sid]
        for frozen in (old,reference_choices[sid]):
            require(s['teacher']['file_sha256']==frozen['teacher_file_sha256'],'teacher choice drift')
            for arm in ('C','J'):
                for field in ('budget','file_sha256','state_sha256'):
                    require(s['arms'][arm][field]==frozen['arms'][arm][field],'selected choice drift')
        old_base=pathlib.PurePosixPath(old['teacher_path']).parent
        require(v['split']['sha256']==reference_manifest[str(old_base/'split_ids.npz')],'split reference drift')
        sources={};refs={}
        for arm in ARMS:
            source=s['teacher'] if arm=='teacher' else s['arms'][arm]
            sources[arm]=verify_file(input_root,{'path':source['path'],'sha256':source['file_sha256']})
            budget=2048 if arm=='teacher' else source['budget']
            if arm!='teacher':require(expected[arm+'_budget']==budget,'budget handoff mismatch')
            require(type(expected[arm+'_parameters']) is int and expected[arm+'_parameters']>0,'parameter census')
            key='teacher_logits' if arm=='teacher' else (arm+'_logits' if budget==2048 else f'{arm}_{budget}_logits')
            require(v['logits'][arm]['key']==key,'original selected logit key')
            old_reference=old_base/'validation_arrays.npz' if budget==2048 else pathlib.PurePosixPath(old['arms'][arm]['path']).parent/'validation_arrays.npz'
            require(v['logits'][arm]['sha256']==reference_manifest[str(old_reference)],'logit reference drift')
            refs[arm]=verify_file(input_root,v['logits'][arm])
            name=f'seed_{sid}_{arm}_stored.npz';p=contained(storage_root,name)
            require(p.stat().st_size==artifact_map[name]['bytes'] and sha(p)==artifact_map[name]['sha256'],'stored archive binding')
        split=verify_file(input_root,v['split'])
        resolved.append({'selection':s,'expected':expected,'validation':v,'sources':sources,'references':refs,'split':split})
    # Deflated siblings are listed/bound by ARTIFACTS but not read or required by latency.
    return {'manifest':manifest,'models':resolved,'artifacts':artifact_map,'storage_root':storage_root,'csv':csv_path}


def package_hashes(here):
    here=pathlib.Path(here)
    mapping=read_json(here/'SOURCE_MAP.json')['export_file_sha256']
    require(type(mapping) is dict and mapping,'source map')
    for name,expected in mapping.items():
        require('/' not in name and relative_path(name)==name,'package path')
        require(sha(here/name)==expected,'source package drift')
    return {**mapping,'SOURCE_MAP.json':sha(here/'SOURCE_MAP.json')}


def validate_release(release,manifest_sha256,source_hashes,head,now):
    exact_keys(release,('schema','purpose','attempt_id','control_root_sha256','deadline_utc','parent_approved','independent_review','no_active_parallel_benchmark','input_manifest_sha256','source_manifest','implementation_commit'),'release')
    require(release['schema']==RELEASE_SCHEMA,'new release schema required')
    require(release['purpose']=='separate-authorized-technical-attempt-not-historical','nonhistorical purpose')
    require(type(release['attempt_id']) is str and re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',release['attempt_id']) is not None,'attempt id')
    require(release['parent_approved'] is True and release['independent_review']=='GO' and release['no_active_parallel_benchmark'] is True,'approval attestations')
    validate_digest(release['control_root_sha256'])
    validate_digest(release['input_manifest_sha256'])
    require(release['input_manifest_sha256']==manifest_sha256,'release input binding')
    require(release['source_manifest']==source_hashes,'release source binding')
    require(type(head) is str and re.fullmatch(r'[0-9a-f]{40}',head) is not None,'commit SHA')
    require(release['implementation_commit']==head,'release commit binding')
    require(type(release['deadline_utc']) is str and release['deadline_utc'].endswith('Z'),'UTC deadline')
    deadline=datetime.datetime.fromisoformat(release['deadline_utc'].replace('Z','+00:00')).timestamp()
    require(deadline>now+10,'insufficient deadline window')
    return deadline


def git_identity(repository):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repository,text=True).strip()
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=repository,text=True).strip(),'clean reviewed checkout required')
    return head


def paths(output,input_root,control_root,package_root,attempt_id):
    """No mutation. A shared control root is fixed by the authorizer, never rotated."""
    inp=pathlib.Path(input_root).resolve(strict=True)
    package=pathlib.Path(package_root).resolve(strict=True)
    control=pathlib.Path(control_root).resolve(strict=True)
    require(inp.is_dir() and control.is_dir(),'existing input/control directories required')
    for a,b in ((control,inp),(control,package)):
        require(not a.is_relative_to(b) and not b.is_relative_to(a),'control overlap')
    out=storage.check_output(output,inp,package)
    require(not out.is_relative_to(control) and not control.is_relative_to(out),'output/control overlap')
    registry=control/('attempt-'+attempt_id)
    require(not registry.exists() and not registry.is_symlink(),'permanent attempt consumed; no retry')
    return out,control,registry
