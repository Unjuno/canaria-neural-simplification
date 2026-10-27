#!/usr/bin/env python3
"""Fresh completed matched/grid -> PRIVATE pending-review storage inputs.

No training, inference, scientific rerun, bootstrap, serialization or automatic
approval. Selected NPZ byte streams are copied exactly, never rewritten. Import
is stdlib-only and inert. This is NOT the historical selected archive route.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SCHEMA = 'canaria.fresh-grid-storage-selection.v1'
REVIEW_SCHEMA = 'canaria.fresh-selection-input-review.v1'
BUDGETS = (512, 1024, 1536, 2048, 3072, 4096, 6144)
NEW_BUDGETS = tuple(b for b in BUDGETS if b != 2048)
ARMS = ('C', 'J')
SEED_FILES = frozenset(
    ['START.json', 'environment.json', 'result.json', 'FILES_SHA256.json', 'validation_arrays.npz']
    + [f'{a}_{b}_{suffix}' for a in ARMS for b in NEW_BUDGETS
       for suffix in ('initial.npz', 'final.npz', 'START.json', 'updates.jsonl')])
ARTIFACT_PATHS = frozenset(
    ['run_contract.json', 'SUMMARY.json', 'FILES_SHA256.json']
    + [f'seed_{i}{s}' for i in range(16) for s in ('.log', '_EXIT.json')]
    + [f'seed_{i}/{name}' for i in range(16) for name in SEED_FILES])
CUSTODY_PATHS = frozenset(['claim.json', 'cohort.json', 'FINISHED.json']
                        + [f'seed_{i}.started.json' for i in range(16)])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    return sha_bytes(Path(path).read_bytes())


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'invalid SHA256')
    return value


def same_json(left, right):
    """JSON value equality without Python's bool/int/float coercions."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(same_json(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(same_json(x, y) for x, y in zip(left, right))
    return left == right


def exact(obj, keys, label):
    require(type(obj) is dict and set(obj) == set(keys), label + ' fields')


def pairs(values):
    out = {}
    for key, value in values:
        require(key not in out, 'duplicate JSON field')
        out[key] = value
    return out


def finite_float(text):
    value = float(text)
    require(math.isfinite(value), 'nonfinite JSON number')
    return value


def read(path):
    raw = Path(path).read_bytes()
    require(len(raw) <= 16 * 1024**2, 'metadata byte ceiling')
    return json.loads(raw, object_pairs_hook=pairs, parse_float=finite_float,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dependencies():
    """Pin every sibling byte before loading only the two stdlib contracts."""
    dep = read(HERE / 'SELECTION_DEPENDENCY.json')
    require(dep['schema'] == 'canaria.fresh-selection-dependency.v1', 'dependency schema')
    expected = {f'{pkg}/{p.name}' for pkg in ('digits_matched_source_20261027', 'digits_joint_budget_source_20261027')
                for p in (HERE.parent / pkg).iterdir() if p.is_file() and p.suffix != '.pyc'}
    expected |= {HERE.name + '/' + name for name in ('storage_codec.py', 'manifest_contract.py')}
    require(set(dep['files']) == expected, 'dependency full source census')
    for rel, pin in dep['files'].items():
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts, 'dependency path')
        path = HERE.parent / rel
        require(not any(part.is_symlink() for part in (path, *path.parents))
                and sha(path) == digest(pin), 'source dependency changed: ' + rel)
    grid = HERE.parent / 'digits_joint_budget_source_20261027'
    return (load_module(grid / 'parent_contract.py', '_fresh_selection_parent'),
            load_module(grid / 'grid_contract.py', '_fresh_selection_grid'), grid)


def adapter_files():
    return {name: sha(HERE / name) for name in
            ('prepare_fresh_selection.py', 'SELECTION_DEPENDENCY.json', 'storage_codec.py', 'manifest_contract.py')}


def strict_tree(root, expected, p):
    """Exact complete bytes, no links, bounded file sizes, strict JSON parsing."""
    root = Path(root)
    total = 0
    for path in root.rglob('*'):
        if path.is_file():
            size = path.stat().st_size
            require(size <= 128 * 1024**2, 'input file byte ceiling')
            total += size
    require(total <= 1024**3, 'input tree byte ceiling')
    result = p.complete_files(root, expected)
    for rel in expected:
        if rel.endswith('.json'):
            read(p.contained(root, rel))
    return result


def validate_input_review(review, parent_sha, rid, grid_files, custody_files):
    exact(review, ('schema', 'purpose', 'matched_descriptor_sha256', 'grid_replication_id',
                   'grid_files', 'grid_custody_files', 'adapter_files',
                   'upstream_provenance_approved', 'independent_review', 'reviewer', 'approval_evidence'), 'input review')
    require(review['schema'] == REVIEW_SCHEMA and review['purpose'] == 'fresh-grid-to-storage-only', 'input review contract')
    require(review['upstream_provenance_approved'] is True and review['independent_review'] == 'GO', 'reviewed upstream provenance required')
    require(all(type(review[k]) is str and review[k].strip() for k in ('reviewer', 'approval_evidence')), 'input review provenance missing')
    require(review['matched_descriptor_sha256'] == parent_sha and review['grid_replication_id'] == rid, 'review input identity')
    require(review['grid_files'] == grid_files and review['grid_custody_files'] == custody_files, 'review full-byte binding')
    require(review['adapter_files'] == adapter_files(), 'review adapter source binding')


def check_grid(grid_workspace, rid, parent, parent_sha, parent_root, grid_root, p, g):
    registry, root = g.context(grid_workspace, rid, grid_root)
    files = strict_tree(root, ARTIFACT_PATHS, p)
    custody = strict_tree(registry, CUSTODY_PATHS, p)
    require(read(root / 'FILES_SHA256.json') == {k: v for k, v in files.items() if k != 'FILES_SHA256.json'}, 'grid root manifest mismatch')
    for i in range(16):
        require(read(root / f'seed_{i}/FILES_SHA256.json') ==
                {name: files[f'seed_{i}/{name}'] for name in SEED_FILES if name != 'FILES_SHA256.json'}, 'grid nested manifest mismatch')
    run, cohort, claim, finished = [read(path) for path in
        (root / 'run_contract.json', registry / 'cohort.json', registry / 'claim.json', registry / 'FINISHED.json')]
    release = cohort.get('release', {})
    # Completion is checked against its existing deadline, never today's clock;
    # validation at the recorded start does not revive or authorize an attempt.
    start = p.timestamp(cohort['utc'])
    g.validate_release(release, grid_root, now_timestamp=start)
    require(g.public_release(release) == run and run['replication_id'] == rid, 'grid private/public release binding')
    require(run['parent_manifest_sha256'] == parent_sha and run['parent_replication_id'] == parent['parent_replication_id']
            and run['matched_package_sha256'] == parent['matched_package_sha256'], 'grid matched parent binding')
    require(cohort.get('status') == 'STARTED' and Path(cohort.get('out', '')).resolve() == root.resolve(), 'grid cohort output binding')
    digest(cohort.get('token_hash'))
    require(claim.get('replication_id') == rid and claim.get('package_sha256') == g.package_identity(grid_root)
            and claim.get('output_relative') == 'artifacts/' + rid, 'grid permanent claim binding')
    summary = read(root / 'SUMMARY.json')
    require(summary.get('decision') in ('CONDITIONAL_GRID_ADVANTAGE', 'CONDITIONAL_GRID_DISADVANTAGE', 'UNCERTAIN', 'CENSORED_NO_COHORT_RATIO'), 'grid incomplete decision')
    require(summary.get('abort_reason') is None and summary.get('not_started') == [], 'grid aborted/not-started')
    attempts = summary.get('attempts', [])
    require(type(attempts) is list and len(attempts) == 16 and {r.get('id') for r in attempts} == set(range(16))
            and all(type(r.get('id')) is int and type(r.get('returncode')) is int and r['returncode'] == 0 and r.get('reason') is None for r in attempts), 'grid successful exit census')
    end = p.timestamp(summary['finished_utc'])
    require(start <= end <= cohort['end_timestamp'] <= p.timestamp(run['deadline_utc']), 'grid completion deadline binding')
    require(finished.get('summary_sha256') == files['SUMMARY.json'] and finished.get('decision') == summary['decision']
            and p.timestamp(finished['utc']) >= end, 'grid completion custody binding')
    rows = summary.get('all_rows', [])
    require(type(rows) is list and len(rows) == 16 and {r.get('id') for r in rows} == set(range(16)), 'grid summary row census')
    seeds = read(grid_root / 'SEEDS.json')['seeds']
    for seed in seeds:
        i = seed['id']
        require(type(next(r for r in rows if r['id'] == i)['id']) is int, 'grid integer teacher identity')
        row = read(root / f'seed_{i}/result.json')
        require(type(row.get('id')) is int and row['id'] == i, 'grid integer result identity')
        require(same_json(row, next(r for r in rows if r['id'] == i)), 'grid summary/result mismatch')
        exit_record = read(root / f'seed_{i}_EXIT.json')
        require(type(exit_record.get('id')) is int and exit_record['id'] == i, 'grid integer exit identity')
        require(same_json(exit_record, next(r for r in attempts if r['id'] == i)), 'grid exit/result mismatch')
        started = read(registry / f'seed_{i}.started.json')
        require(type(started.get('id')) is int and started['id'] == i, 'grid integer admission identity')
        admission = read(root / f'seed_{i}/START.json')
        require(type(admission.get('id')) is int and admission['id'] == i, 'grid integer START identity')
        require(same_json(admission.get('seed'), seed) and admission.get('files') == g.manifest_hashes(grid_root)
                and admission.get('package_sha256') == g.package_identity(grid_root), 'grid seed source binding')
        old = read(parent_root / f'seed_{i}/result.json')
        require(row.get('id') == i and row.get('status') == 'COMPLETE', 'grid result identity/status')
        require(row.get('parent_result_sha256') == parent['files'][f'seed_{i}/result.json']
                and row.get('parent_teacher_initial_sha256') == old['teacher_initial_sha256']
                and row.get('parent_teacher_final_sha256') == old['teacher_final_sha256']
                and row.get('schedule_sha256') == old['schedule_sha256']
                and row.get('teacher_correct') == old['teacher_correct'], 'grid teacher/schedule/result parent binding')
        require(set(row.get('fit_records', {})) == {f'{a}_{b}' for a in ARMS for b in NEW_BUDGETS}, 'grid fit census')
        for arm in ARMS:
            for budget in NEW_BUDGETS:
                key = f'{arm}_{budget}'
                fit = row['fit_records'][key]
                require(fit.get('schedule_sha256') == old['schedule_sha256'] and fit.get('parameter_steps') == [600] * (2 if arm == 'C' else 4)
                        and fit.get('examples_per_parameter') == 76800, 'grid fixed fit contract')
                cell_start = read(root / f'seed_{i}/{key}_START.json')
                require(cell_start.get('arm') == arm and cell_start.get('budget') == budget
                        and cell_start.get('schedule_sha256') == old['schedule_sha256'], 'grid fit admission binding')
                digest(cell_start.get('initial_state_sha256'))
        cells = row.get('cells', [])
        require(type(cells) is list and len(cells) == 14 and
                {(c.get('arm'), c.get('budget')) for c in cells} == {(a, b) for a in ARMS for b in BUDGETS}, 'grid 224-cell census')
        for cell in cells:
            arm, budget = cell['arm'], cell['budget']
            require(type(budget) is int and type(cell.get('reused')) is bool and cell['reused'] == (budget == 2048), 'grid reuse identity')
            nmse, correct = cell.get('nmse'), cell.get('correct')
            require(type(nmse) in (int, float) and math.isfinite(nmse) and nmse >= 0
                    and type(correct) is int and 0 <= correct <= 270 and type(cell.get('n')) is int and cell['n'] == 270,
                    'grid scalar types/ranges')
            actual = nmse <= .08 and correct / 270 >= old['teacher_correct'] / 270 - .02
            require(cell.get('accuracy') == correct / 270 and type(cell.get('quality_pass')) is bool
                    and cell['quality_pass'] == actual, 'grid pass/accuracy mismatch')
            digest(cell.get('state_sha256'))
            if budget == 2048:
                require(cell == dict(old['arms'][arm], arm=arm, budget=2048, reused=True), 'grid reused 2048 cell changed')
    minima, patterns, pass_counts, coverage = selection_metadata(rows)
    for key, value in (('minima', minima), ('patterns', patterns), ('pass_counts', pass_counts), ('coverage', coverage)):
        require(summary.get(key) == value, 'grid ' + key + ' mismatch')
    if coverage['both'] != 16:
        require(summary['decision'] == 'CENSORED_NO_COHORT_RATIO', 'grid censoring decision mismatch')
        raise ValueError('censored cohort: no storage selection; no dropping or substitution')
    require(summary['decision'] != 'CENSORED_NO_COHORT_RATIO', 'grid uncensored decision mismatch')
    # Bootstrap/scientific decision is not rerun or used to choose weights. Its
    # original exact bytes remain bound in summary/file/review digests.
    return root, registry, files, custody, summary, minima, patterns


def selection_metadata(rows):
    minima, patterns = [], []
    counts = {a: {str(b): 0 for b in BUDGETS} for a in ARMS}
    for row in sorted(rows, key=lambda x: x['id']):
        choice = {'id': row['id']}
        for arm in ARMS:
            flags = {c['budget']: c['quality_pass'] for c in row['cells'] if c['arm'] == arm}
            passing = [b for b in BUDGETS if flags[b]]
            choice[arm] = passing[0] if passing else None
            for b in BUDGETS:
                counts[arm][str(b)] += int(flags[b])
            patterns.append({'id': row['id'], 'arm': arm, 'pass_by_budget': [flags[b] for b in BUDGETS],
                             'nonmonotone': any(flags[BUDGETS[i]] and not flags[BUDGETS[j]] for i in range(7) for j in range(i + 1, 7))})
        minima.append(choice)
    coverage = {'both': sum(x['C'] is not None and x['J'] is not None for x in minima),
                'C_only': sum(x['C'] is not None and x['J'] is None for x in minima),
                'J_only': sum(x['C'] is None and x['J'] is not None for x in minima),
                'neither': sum(x['C'] is None and x['J'] is None for x in minima)}
    return minima, patterns, counts, coverage


def target_path(output, protected):
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), 'output must be fresh and absent')
    for path in (output.parent, *output.parent.parents):
        require(not path.is_symlink(), 'output ancestor symlink')
    parent = output.parent.resolve(strict=True)
    require(parent.is_dir(), 'existing output parent required')
    target = parent / output.name
    for path in protected:
        path = Path(path).resolve(strict=True)
        require(not target.is_relative_to(path) and not path.is_relative_to(target), 'output overlaps protected input/source')
    return target


def prepare(matched_workspace, matched_descriptor, grid_workspace, grid_replication_id,
            input_review, input_review_sha256, output):
    """Requires separate authorization before use on actual completed outputs."""
    protected = (matched_workspace, grid_workspace, HERE.parent, matched_descriptor, input_review)
    target = target_path(output, protected)  # before artifact reads or NumPy import
    p, g, grid_package = dependencies()
    for path in (Path(matched_descriptor), Path(input_review)):
        require(path.is_file() and not path.is_symlink(), 'regular nonsymlink descriptor/review required')
    parent_raw = Path(matched_descriptor).read_bytes()
    parent_sha = sha_bytes(parent_raw)
    parent = read(matched_descriptor)
    # Strengthen legacy JSON readers without changing the upstream contracts.
    parent_root, parent_custody = p.parent_paths(matched_workspace, parent['parent_replication_id'])
    strict_tree(parent_root, p.ARTIFACT_PATHS, p)
    strict_tree(parent_custody, p.CUSTODY_PATHS, p)
    p.verify_parent(matched_workspace, parent, grid_package)
    matched_summary = read(parent_root / 'SUMMARY.json')
    require(all(type(row.get('id')) is int for row in matched_summary['all_rows'] + matched_summary['attempts']),
            'matched integer teacher identity')
    for i in range(16):
        require(type(read(parent_root / f'seed_{i}/result.json').get('id')) is int
                and type(read(parent_custody / f'seed_{i}.started.json').get('id')) is int,
                'matched integer admission/result identity')
    require(not Path(grid_workspace).resolve().is_relative_to(Path(matched_workspace).resolve())
            and not Path(matched_workspace).resolve().is_relative_to(Path(grid_workspace).resolve()), 'separate grid/matched custody roots required')
    root, registry, files, custody, summary, minima, patterns = check_grid(
        grid_workspace, grid_replication_id, parent, parent_sha, parent_root, grid_package, p, g)
    require(sha(input_review) == digest(input_review_sha256), 'input review SHA256 mismatch')
    review = read(input_review)
    validate_input_review(review, parent_sha, grid_replication_id, files, custody)
    codec = load_module(HERE / 'storage_codec.py', '_fresh_selection_codec')
    manifest_contract = load_module(HERE / 'manifest_contract.py', '_fresh_selection_manifest')
    models, source_map, payloads = [], [], {}
    for choice in minima:
        i = choice['id']
        old = read(parent_root / f'seed_{i}/result.json')
        row = next(r for r in summary['all_rows'] if r['id'] == i)
        model = {'id': i, 'teacher': None, 'arms': {}}
        for arm in ('teacher', *ARMS):
            budget = None if arm == 'teacher' else choice[arm]
            from_matched = arm == 'teacher' or budget == 2048
            relative = f'seed_{i}/' + ('teacher_final.npz' if arm == 'teacher' else f'{arm}_final.npz' if from_matched else f'{arm}_{budget}_final.npz')
            origin = parent_root if from_matched else root
            pin = (parent['files'] if from_matched else files)[relative]
            path = p.contained(origin, relative)
            raw = path.read_bytes()
            require(sha_bytes(raw) == pin, 'selected source changed')
            shapes = codec.shapes_teacher() if arm == 'teacher' else codec.shapes_arm(arm, budget)
            arrays = codec.load_source(path, pin, shapes)
            logical = codec.state_hash(arrays)
            expected = old['teacher_final_sha256'] if arm == 'teacher' else next(c for c in row['cells'] if c['arm'] == arm and c['budget'] == budget)['state_sha256']
            require(logical == expected, 'selected teacher/arm logical state mismatch')
            dest = f'inputs/seed_{i}_{arm}.npz'
            payloads[dest] = raw
            selected = {'path': dest, 'file_sha256': pin}
            if arm == 'teacher':
                model['teacher'] = selected
            else:
                selected.update(state_sha256=logical, budget=budget)
                model['arms'][arm] = selected
            source_map.append({'id': i, 'arm': arm, 'budget': budget,
                               'source_stage': 'matched' if from_matched else 'grid', 'source_path': relative,
                               'destination_path': dest, 'file_sha256': pin, 'state_sha256': logical})
        models.append(model)
    require(len(payloads) == 48 and sum(map(len, payloads.values())) < 128 * 1024**2, '48-source byte census')
    # Verify upstream snapshots again before any output is created. This is a
    # local custody check, not a cryptographic attestation of numerical science.
    require(p.validate_completed_parent(matched_workspace, parent['parent_replication_id'], grid_package) == parent, 'matched changed during preparation')
    require(strict_tree(root, ARTIFACT_PATHS, p) == files and strict_tree(registry, CUSTODY_PATHS, p) == custody,
            'grid changed during preparation')
    require(sha(matched_descriptor) == parent_sha and sha(input_review) == input_review_sha256, 'review/descriptor changed during preparation')
    dependencies()
    require(review['adapter_files'] == adapter_files(), 'adapter changed during preparation')
    origin = {'schema': SCHEMA, 'evidence_class': 'fresh-replication-selection-not-historical-archive',
              'historical_attempt': 'CLOSED_NOT_REOPENED', 'status': 'PENDING_CUSTODIAN_REVIEW',
              'storage_execution_authorized': False, 'adapter_files': adapter_files(),
              'matched_descriptor_sha256': parent_sha, 'matched_descriptor': parent,
              'grid_replication_id': grid_replication_id, 'grid_files': files, 'grid_custody_files': custody,
              'input_review_sha256': input_review_sha256, 'grid_summary': summary,
              'selection_rule': 'smallest passing observed budget; no interpolation, no monotonicity assumption',
              'cohort_size': 16, 'cell_count': 224, 'minima': minima, 'patterns': patterns,
              'source_map': source_map}
    origin_bytes = encoded(origin)
    manifest = {'schema': manifest_contract.SCHEMA, 'cohort_size': 16,
                'selection_origin_sha256': sha_bytes(origin_bytes), 'models': models}
    manifest_contract.validate_manifest(manifest)
    manifest_bytes = encoded(manifest)
    target.mkdir(exist_ok=False)
    (target / 'inputs').mkdir()
    for rel, raw in payloads.items():
        with (target / rel).open('xb') as stream:
            stream.write(raw)
        require(sha(target / rel) == sha_bytes(raw), 'copied source readback mismatch')
    for name, raw in (('FRESH_SELECTION.json', origin_bytes), ('selected_models.json', manifest_bytes)):
        with (target / name).open('xb') as stream:
            stream.write(raw)
    gate = {'schema': 'canaria.fresh-selection-custodian-gate.v1', 'status': 'PENDING_CUSTODIAN_REVIEW',
            'storage_execution_authorized': False, 'selection_origin_sha256': sha_bytes(origin_bytes),
            'manifest_sha256': sha_bytes(manifest_bytes), 'source_count': 48,
            'required_next_step': 'Independent custodian compares all 16 teachers/32 choices and exact source bindings; separately authorize storage execution.'}
    with (target / 'REVIEW_REQUIRED.json').open('xb') as stream:
        stream.write(encoded(gate))
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matched-workspace', type=Path, required=True)
    parser.add_argument('--matched-descriptor', type=Path, required=True)
    parser.add_argument('--grid-workspace', type=Path, required=True)
    parser.add_argument('--grid-replication-id', required=True)
    parser.add_argument('--input-review', type=Path, required=True)
    parser.add_argument('--input-review-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
