"""Fresh selected-storage handoff, never historical-v1 provenance substitution.

Import and metadata verification are stdlib-only, read-only. Numerical payload
checks are lazy and used only inside a separately authorized worker. No reference
arrays, approval, selection, output directory or scientific attempt is created.
"""
from __future__ import annotations
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import zipfile
import latency_contract as historical

HERE = Path(__file__).resolve().parent
SCHEMA = 'canaria.selected-latency-fresh-inputs.v1'
RELEASE_SCHEMA = 'canaria.selected-latency-fresh-release.v1'
REVIEW_SCHEMA = 'canaria.selected-latency-fresh-review.v1'
ARMS = historical.ARMS
CSV_SHA256 = historical.CSV_SHA256
STORAGE_FILES = ('ENV.json', 'ROWS.json', 'CENSUS.json', 'ARTIFACTS.json',
                 'SOURCES.json', 'TENSORS.json', 'SUMMARY.json', 'ROWS.csv')
require = historical.require
exact_keys = historical.exact_keys
validate_digest = historical.validate_digest
relative_path = historical.relative_path
descriptor = historical.descriptor
sha = historical.sha
paths = historical.paths
git_identity = historical.git_identity


def same(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(same(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(same(x, y) for x, y in zip(left, right))
    return left == right


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def read_json(path, expected=None):
    raw = Path(path).read_bytes()
    require(len(raw) <= 16 * 1024**2, 'fresh metadata byte ceiling')
    if expected is not None:
        require(hashlib.sha256(raw).hexdigest() == validate_digest(expected), 'fresh metadata SHA256 mismatch')
    def finite(text):
        value = float(text)
        require(math.isfinite(value), 'nonfinite JSON number')
        return value
    return json.loads(raw, object_pairs_hook=historical.storage.no_duplicate_fields,
                      parse_float=finite,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def contained(root, relative, directory=False):
    root = Path(root).absolute()
    require(not any(p.is_symlink() for p in (root, *root.parents)), 'input root ancestor symlink')
    return historical.contained(root, relative, directory)


def verify_file(root, value):
    path = contained(root, value['path'])
    require(path.stat().st_size <= 128 * 1024**2, 'input file byte ceiling')
    require(sha(path) == value['sha256'], 'fresh input SHA256 mismatch')
    return path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    dep = read_json(HERE / 'FRESH_DEPENDENCY.json')
    exact_keys(dep, ('schema', 'base_public_commit', 'files'), 'fresh dependency')
    require(dep['schema'] == 'canaria.fresh-latency-dependency.v1', 'dependency schema')
    packages = ('digits_matched_source_20261027', 'digits_joint_budget_source_20261027',
                'selected_whole_storage_source_20261027')
    expected = {pkg + '/' + p.name for pkg in packages for p in (HERE.parent / pkg).iterdir()
                if p.is_file() and p.suffix != '.pyc'}
    require(set(dep['files']) == expected, 'fresh complete dependency census')
    for rel, pin in dep['files'].items():
        require(sha(contained(HERE.parent, rel)) == validate_digest(pin), 'fresh source dependency drift')
    selection = HERE.parent / 'selected_whole_storage_source_20261027'
    adapter = load_module(selection / 'prepare_fresh_selection.py', '_latency_fresh_selection')
    p, g, grid = adapter.dependencies()  # stdlib contracts only, never runners
    return adapter, p, g, grid, selection


def package_hashes(here):
    result = historical.package_hashes(here)
    dependencies()
    return result


def validate_release(release, manifest_sha256, source_hashes, head, now):
    # Only common lifecycle checks are shared; input/provenance schemas are never
    # translated into historical input v1 and none of its pins is modified.
    require(type(release) is dict and release.get('schema') == RELEASE_SCHEMA, 'fresh release schema required')
    return historical.validate_release(dict(release, schema=historical.RELEASE_SCHEMA),
                                       manifest_sha256, source_hashes, head, now)


def basis_sha256(manifest):
    return hashlib.sha256(encoded({k: v for k, v in manifest.items() if k != 'input_review'})).hexdigest()


def validate_inputs(manifest):
    exact_keys(manifest, ('schema', 'selection_directory', 'selection', 'fresh_selection', 'pending_gate',
                         'parent_descriptor', 'upstream_review', 'input_review', 'matched_directory',
                         'grid_directory', 'grid_summary', 'storage', 'csv', 'validation'), 'fresh latency inputs')
    require(manifest['schema'] == SCHEMA, 'fresh input schema')
    for k in ('selection_directory', 'matched_directory', 'grid_directory'):
        relative_path(manifest[k])
    for k in ('selection', 'fresh_selection', 'pending_gate', 'parent_descriptor',
              'upstream_review', 'input_review', 'grid_summary', 'csv'):
        descriptor(manifest[k])
    require(manifest['csv']['sha256'] == CSV_SHA256, 'frozen CSV identity')
    exact_keys(manifest['storage'], ('directory', 'files'), 'fresh storage')
    relative_path(manifest['storage']['directory'])
    exact_keys(manifest['storage']['files'], STORAGE_FILES, 'complete storage metadata')
    for name, entry in manifest['storage']['files'].items():
        descriptor(entry)
        require(entry['path'] == manifest['storage']['directory'] + '/' + name, 'storage metadata path')
    for key, name in (('selection', 'selected_models.json'), ('fresh_selection', 'FRESH_SELECTION.json'),
                      ('pending_gate', 'REVIEW_REQUIRED.json')):
        require(manifest[key]['path'] == manifest['selection_directory'] + '/' + name, 'selection bundle path')
    require(manifest['grid_summary']['path'] == manifest['grid_directory'] + '/SUMMARY.json', 'grid summary path')
    rows = manifest['validation']
    require(type(rows) is list and len(rows) == 16, 'validation census')
    ids = []
    for row in rows:
        exact_keys(row, ('id', 'split', 'logits'), 'validation row')
        require(type(row['id']) is int and 0 <= row['id'] < 16, 'validation id')
        ids.append(row['id']); descriptor(row['split']); exact_keys(row['logits'], ARMS, 'logit census')
        for arm in ARMS:
            descriptor(row['logits'][arm], ('key',))
            require(type(row['logits'][arm]['key']) is str, 'logit key type')
    require(ids == list(range(16)), 'ordered unique validation census')
    return manifest


def validation_descriptors(origin, models, matched_directory, grid_directory):
    """Exact already-saved producer references; no generated or inferred logits."""
    rows = []
    for s in sorted(models, key=lambda r: r['id']):
        i = s['id']; split = f'seed_{i}/split_ids.npz'
        row = {'id': i, 'split': {'path': matched_directory + '/' + split,
                                 'sha256': origin['matched_descriptor']['files'][split]}, 'logits': {}}
        for arm in ARMS:
            budget = 2048 if arm == 'teacher' else s['arms'][arm]['budget']
            matched = budget == 2048
            directory = matched_directory if matched else grid_directory
            pins = origin['matched_descriptor']['files'] if matched else origin['grid_files']
            rel = f'seed_{i}/validation_arrays.npz'
            key = 'teacher_logits' if arm == 'teacher' else arm + '_logits' if matched else f'{arm}_{budget}_logits'
            row['logits'][arm] = {'path': directory + '/' + rel, 'sha256': pins[rel], 'key': key}
        rows.append(row)
    return rows


def check_origin(origin, selection, parent, upstream, summary, a, p, g, grid):
    exact_keys(origin, ('schema', 'evidence_class', 'historical_attempt', 'status', 'storage_execution_authorized',
                        'adapter_files', 'matched_descriptor_sha256', 'matched_descriptor', 'grid_replication_id',
                        'grid_files', 'grid_custody_files', 'input_review_sha256', 'grid_summary', 'selection_rule',
                        'cohort_size', 'cell_count', 'minima', 'patterns', 'source_map'), 'fresh selection provenance')
    require(origin['schema'] == a.SCHEMA and origin['evidence_class'] == 'fresh-replication-selection-not-historical-archive'
            and origin['historical_attempt'] == 'CLOSED_NOT_REOPENED', 'not fresh selection provenance')
    require(origin['status'] == 'PENDING_CUSTODIAN_REVIEW' and origin['storage_execution_authorized'] is False,
            'immutable pending selection record changed')
    require(same(origin['adapter_files'], a.adapter_files()), 'selection adapter identity')
    require(type(origin['cohort_size']) is int and origin['cohort_size'] == 16
            and type(origin['cell_count']) is int and origin['cell_count'] == 224, 'selection census')
    require(origin['selection_rule'] == 'smallest passing observed budget; no interpolation, no monotonicity assumption', 'selection rule')
    exact_keys(parent, ('contract', 'parent_contract', 'historical_attempt', 'parent_replication_id',
                        'matched_package_sha256', 'files', 'custody_files'), 'parent descriptor')
    _, dep = p.dependency(grid)
    require(parent['contract'] == p.CONTRACT and parent['parent_contract'] == p.MATCHED_CONTRACT
            and parent['historical_attempt'] == p.CLOSED and parent['matched_package_sha256'] == dep['matched_package_sha256'], 'fresh parent identity')
    p.identity(parent['parent_replication_id']); p.identity(origin['grid_replication_id'])
    p.digest_map(parent['files'], p.ARTIFACT_PATHS); p.digest_map(parent['custody_files'], p.CUSTODY_PATHS)
    p.digest_map(origin['grid_files'], a.ARTIFACT_PATHS); p.digest_map(origin['grid_custody_files'], a.CUSTODY_PATHS)
    require(same(parent, origin['matched_descriptor']), 'embedded parent mismatch')
    a.validate_input_review(upstream, origin['matched_descriptor_sha256'], origin['grid_replication_id'],
                            origin['grid_files'], origin['grid_custody_files'])
    require(same(summary, origin['grid_summary']), 'original grid summary mismatch')
    require(summary.get('decision') in ('CONDITIONAL_GRID_ADVANTAGE', 'CONDITIONAL_GRID_DISADVANTAGE', 'UNCERTAIN')
            and summary.get('abort_reason') is None and summary.get('not_started') == [], 'incomplete/censored grid')
    rows = summary.get('all_rows')
    require(type(rows) is list and len(rows) == 16 and all(type(r.get('id')) is int for r in rows)
            and sorted(r['id'] for r in rows) == list(range(16)), 'grid row census')
    for row in rows:
        require(row.get('status') == 'COMPLETE' and type(row.get('teacher_correct')) is int
                and 0 <= row['teacher_correct'] <= 270, 'grid row completion/count')
        cells = row.get('cells')
        require(type(cells) is list and len(cells) == 14, 'grid cell census')
        observed = set()
        for cell in cells:
            arm, budget = cell.get('arm'), cell.get('budget')
            require(type(arm) is str and arm in ('C', 'J') and type(budget) is int and budget in a.BUDGETS,
                    'grid cell identity')
            require((arm, budget) not in observed, 'duplicate grid cell'); observed.add((arm, budget))
            require(type(cell.get('reused')) is bool and cell['reused'] == (budget == 2048), 'grid reuse identity')
            nmse, correct = cell.get('nmse'), cell.get('correct')
            require(type(nmse) in (int, float) and math.isfinite(nmse) and nmse >= 0
                    and type(correct) is int and 0 <= correct <= 270 and type(cell.get('n')) is int
                    and cell['n'] == 270, 'grid scalar types')
            require(type(cell.get('quality_pass')) is bool and cell['quality_pass'] ==
                    (nmse <= .08 and correct / 270 >= row['teacher_correct'] / 270 - .02)
                    and cell.get('accuracy') == correct / 270, 'grid pass binding')
            validate_digest(cell.get('state_sha256'))
    minima, patterns, counts, coverage = a.selection_metadata(rows)
    require(coverage['both'] == 16, 'censored selection refused')
    for key, expected in (('minima', minima), ('patterns', patterns), ('pass_counts', counts), ('coverage', coverage)):
        require(same(summary.get(key), expected), 'grid selected summary mismatch: ' + key)
    require(same(origin['minima'], minima) and same(origin['patterns'], patterns), 'origin selection mismatch')
    source_map = origin['source_map']
    require(type(source_map) is list and len(source_map) == 48, '48 source-map entries required')
    mapping = {}; model_map = {s['id']: s for s in selection['models']}
    for entry in source_map:
        exact_keys(entry, ('id', 'arm', 'budget', 'source_stage', 'source_path', 'destination_path', 'file_sha256', 'state_sha256'), 'selected source mapping')
        i, arm = entry['id'], entry['arm']
        require(type(i) is int and i in range(16) and type(arm) is str and arm in ARMS
                and (i, arm) not in mapping, 'selected source mapping identity')
        s = model_map[i]; source = s['teacher'] if arm == 'teacher' else s['arms'][arm]
        budget = None if arm == 'teacher' else source['budget']
        require(same(entry['budget'], budget), 'selected mapping budget')
        if arm != 'teacher':
            require(budget == minima[i][arm], 'first observed selection drift')
        matched = arm == 'teacher' or budget == 2048
        rel = f'seed_{i}/' + ('teacher_final.npz' if arm == 'teacher' else f'{arm}_final.npz' if matched else f'{arm}_{budget}_final.npz')
        require(entry['source_stage'] == ('matched' if matched else 'grid') and entry['source_path'] == rel,
                'selected source routing drift')
        require(entry['destination_path'] == source['path'] == f'inputs/seed_{i}_{arm}.npz', 'selected destination drift')
        pins = parent['files'] if matched else origin['grid_files']
        require(entry['file_sha256'] == source['file_sha256'] == pins[rel], 'selected file hash drift')
        grid_row = next(r for r in rows if r['id'] == i)
        expected = grid_row['parent_teacher_final_sha256'] if arm == 'teacher' else next(
            c['state_sha256'] for c in grid_row['cells'] if c['arm'] == arm and c['budget'] == budget)
        require(validate_digest(entry['state_sha256']) == expected, 'selected logical state drift')
        if arm != 'teacher':
            require(source['state_sha256'] == expected, 'selection logical state drift')
        mapping[i, arm] = entry
    return mapping


def check_review(review, manifest):
    exact_keys(review, ('schema', 'purpose', 'input_basis_sha256', 'source_manifest', 'selection_review',
                        'storage_execution_approved', 'storage_complete_review', 'reference_provenance_review',
                        'independent_review', 'reviewer', 'approval_evidence'), 'fresh handoff review')
    require(review['schema'] == REVIEW_SCHEMA and review['purpose'] == 'fresh-storage-to-latency-only', 'fresh review purpose')
    require(all(review[k] == 'GO' for k in ('selection_review', 'storage_complete_review', 'reference_provenance_review', 'independent_review'))
            and review['storage_execution_approved'] is True, 'independent selection/storage/reference review required')
    require(all(type(review[k]) is str and review[k].strip() for k in ('reviewer', 'approval_evidence')), 'review provenance missing')
    require(review['input_basis_sha256'] == basis_sha256(manifest), 'review input basis binding')
    require(same(review['source_manifest'], package_hashes(HERE)), 'review fresh source binding')


def load_metadata(manifest, input_root):
    """Read-only full-byte/metadata handoff; never creates valid model references."""
    validate_inputs(manifest)
    a, p, g, grid, storage_package = dependencies()
    review = read_json(verify_file(input_root, manifest['input_review']))
    check_review(review, manifest)  # before any source/archive content checks
    docs = {key: read_json(verify_file(input_root, manifest[key])) for key in
            ('selection', 'fresh_selection', 'pending_gate', 'parent_descriptor', 'upstream_review', 'grid_summary')}
    selection = historical.storage.validate_manifest(docs['selection']); origin = docs['fresh_selection']
    require(selection['selection_origin_sha256'] == manifest['fresh_selection']['sha256'], 'fresh origin digest binding')
    require(origin['matched_descriptor_sha256'] == manifest['parent_descriptor']['sha256']
            and origin['input_review_sha256'] == manifest['upstream_review']['sha256']
            and origin['grid_files']['SUMMARY.json'] == manifest['grid_summary']['sha256'], 'upstream record digest binding')
    gate = docs['pending_gate']
    exact_keys(gate, ('schema', 'status', 'storage_execution_authorized', 'selection_origin_sha256',
                     'manifest_sha256', 'source_count', 'required_next_step'), 'pending selection gate')
    require(gate['schema'] == 'canaria.fresh-selection-custodian-gate.v1'
            and gate['status'] == 'PENDING_CUSTODIAN_REVIEW' and gate['storage_execution_authorized'] is False
            and gate['selection_origin_sha256'] == manifest['fresh_selection']['sha256']
            and gate['manifest_sha256'] == manifest['selection']['sha256']
            and type(gate['source_count']) is int and gate['source_count'] == 48, 'immutable selection gate binding')
    mapping = check_origin(origin, selection, docs['parent_descriptor'], docs['upstream_review'], docs['grid_summary'], a, p, g, grid)
    expected_validation = validation_descriptors(origin, selection['models'], manifest['matched_directory'], manifest['grid_directory'])
    require(same(manifest['validation'], expected_validation), 'saved validation reference binding')
    store = manifest['storage']; storage_root = contained(input_root, store['directory'], directory=True)
    data = {name: read_json(verify_file(input_root, desc)) if name.endswith('.json') else verify_file(input_root, desc)
            for name, desc in store['files'].items()}
    env = data['ENV.json']
    require(env.get('manifest_sha256') == manifest['selection']['sha256'], 'storage selection-manifest binding')
    require(same(env.get('source_sha256'), {p.name: sha(p) for p in sorted(storage_package.glob('*.py'))}), 'storage producer source binding')
    summary = data['SUMMARY.json']; census = data['CENSUS.json']
    require(summary.get('status') == 'COMPLETE' and all(type(summary.get(k)) is int and summary[k] == v
            for k, v in (('complete', 16), ('cohort', 16), ('expected_models', 48), ('expected_archives', 96), ('produced_archives', 96)))
            and summary.get('failures') == [], 'storage incomplete cohort')
    require(type(census) is list and len(census) == 16 and all(type(r.get('id')) is int and r.get('status') == 'PASS' for r in census)
            and sorted(r['id'] for r in census) == list(range(16)), 'storage complete census')
    artifacts = data['ARTIFACTS.json']; artifact_map = {}
    names = {f'seed_{i}_{arm}_{label}.npz' for i in range(16) for arm in ARMS for label in ('stored', 'deflated')}
    require(type(artifacts) is list and len(artifacts) == 96, 'stored/deflated metadata census')
    for item in artifacts:
        exact_keys(item, ('file', 'bytes', 'sha256'), 'storage artifact')
        require(item['file'] in names and item['file'] not in artifact_map, 'storage archive identity')
        validate_digest(item['sha256']); require(type(item['bytes']) is int and item['bytes'] > 0, 'archive size')
        artifact_map[item['file']] = item
    require(sum(x['bytes'] for x in artifacts) == summary.get('artifact_bytes') < 128 * 1024**2, 'storage total bytes')
    rows = data['ROWS.json']
    require(type(rows) is list and len(rows) == 16 and all(type(r.get('id')) is int for r in rows)
            and sorted(r['id'] for r in rows) == list(range(16)), 'storage row census')
    row_map = {r['id']: r for r in rows}; resolved = []; expected_sources = []
    selection_root = contained(input_root, manifest['selection_directory'], directory=True)
    for s in sorted(selection['models'], key=lambda r: r['id']):
        i = s['id']; row = row_map[i]; v = manifest['validation'][i]; sources = {}; references = {}
        for arm in ARMS:
            source = s['teacher'] if arm == 'teacher' else s['arms'][arm]
            sources[arm] = verify_file(selection_root, {'path': source['path'], 'sha256': source['file_sha256']})
            entry = {'id': i, 'arm': arm, 'path': source['path'], 'sha256': source['file_sha256']}
            if arm != 'teacher':
                require(type(row.get(arm + '_budget')) is int and row[arm + '_budget'] == source['budget'], 'storage selected budget')
                entry['logical_sha256'] = source['state_sha256']
            expected_sources.append(entry)
            for metric in ('parameters', 'tensors', 'payload_bytes', 'stored_bytes', 'deflated_bytes'):
                require(type(row.get(arm + '_' + metric)) is int and row[arm + '_' + metric] > 0, 'storage row metric type')
            for label in ('stored', 'deflated'):
                name = f'seed_{i}_{arm}_{label}.npz'
                require(row[arm + '_' + label + '_bytes'] == artifact_map[name]['bytes'], 'row/archive size binding')
            name = f'seed_{i}_{arm}_stored.npz'; path = contained(storage_root, name)
            require(path.stat().st_size == artifact_map[name]['bytes'] and sha(path) == artifact_map[name]['sha256'], 'stored archive byte binding')
            references[arm] = verify_file(input_root, v['logits'][arm])
        resolved.append({'selection': s, 'expected': row, 'validation': v, 'sources': sources,
                         'references': references, 'split': verify_file(input_root, v['split'])})
    require(same(data['SOURCES.json'], expected_sources), 'storage selected source/state census')
    csv = verify_file(input_root, manifest['csv'])  # frozen bytes; no fallback or generated substitute
    return {'manifest': manifest, 'models': resolved, 'artifacts': artifact_map, 'storage_root': storage_root,
            'csv': csv, 'source_map': mapping, 'tensors': data['TENSORS.json'], 'storage_package': storage_package}


def verify_payloads(inputs):
    """Lazy NumPy only: exact source/whole-model content, never forward/inference.
    The deployment model loader and 270-row numerical correctness test stay exact.
    """
    codec = load_module(inputs['storage_package'] / 'storage_codec.py', '_fresh_latency_codec')
    np = codec.np; expected_tensors = []
    def archive(path, pin):
        raw = Path(path).read_bytes()
        require(len(raw) <= 128 * 1024**2 and hashlib.sha256(raw).hexdigest() == pin, 'saved reference byte binding')
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            members = z.infolist()
            require(len(members) <= 32 and len({m.filename for m in members}) == len(members)
                    and sum(m.file_size for m in members) <= 32 * 1024**2, 'saved reference expanded-byte census')
        return np.load(io.BytesIO(raw), allow_pickle=False)
    for item in inputs['models']:
        s = item['selection']; i = s['id']; expected = item['expected']
        teacher = codec.load_source(item['sources']['teacher'], s['teacher']['file_sha256'], codec.shapes_teacher())
        require(codec.state_hash(teacher) == inputs['source_map'][i, 'teacher']['state_sha256'], 'teacher logical state binding')
        whole_models = {'teacher': teacher}
        for arm in ('C', 'J'):
            source = s['arms'][arm]
            replacement = codec.load_source(item['sources'][arm], source['file_sha256'], codec.shapes_arm(arm, source['budget']))
            require(codec.state_hash(replacement) == source['state_sha256'], 'replacement logical state binding')
            whole_models[arm] = codec.assemble(teacher, replacement, arm, source['budget'])
        for arm, arrays in whole_models.items():
            name = f'seed_{i}_{arm}_stored.npz'
            saved = codec.load_source(inputs['storage_root'] / name, inputs['artifacts'][name]['sha256'], {k: v.shape for k, v in arrays.items()})
            require(all(saved[k].tobytes() == v.tobytes() for k, v in arrays.items()), 'stored whole-model content mismatch')
            for metric, actual in (('parameters', sum(a.size for a in arrays.values())), ('tensors', len(arrays)),
                                   ('payload_bytes', sum(a.nbytes for a in arrays.values()))):
                require(expected[arm + '_' + metric] == actual, 'storage parameter/payload census')
            for key, array in sorted(arrays.items()):
                expected_tensors.append({'id': i, 'arm': arm, 'destination': key,
                    'source_key': key.removeprefix('replacement.'), 'source_arm': arm if key.startswith('replacement.') else 'teacher',
                    'shape': list(array.shape), 'dtype': array.dtype.str, 'sha256': hashlib.sha256(array.tobytes()).hexdigest()})
        with archive(item['split'], item['validation']['split']['sha256']) as z:
            require(len(z.files) == len(set(z.files)) and 'validation' in z.files, 'saved split keys')
            ids = z['validation']
            require(ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer) and len(ids) == 270
                    and len(set(ids)) == 270 and np.all((0 <= ids) & (ids < 1797)), 'saved validation split')
        for arm in ARMS:
            with archive(item['references'][arm], item['validation']['logits'][arm]['sha256']) as z:
                key = item['validation']['logits'][arm]['key']
                require(len(z.files) == len(set(z.files)) and key in z.files, 'missing/duplicate saved logit key')
                ref = z[key]
                require(ref.dtype == np.dtype('<f4') and ref.shape == (270, 10) and np.isfinite(ref).all(), 'saved logit shape/dtype/finiteness')
    require(same(inputs['tensors'], expected_tensors), 'storage tensor-content provenance')


def load_inputs(manifest_path, expected_sha256, input_root):
    inputs = load_metadata(read_json(manifest_path, expected_sha256), input_root)
    verify_payloads(inputs)
    return inputs
