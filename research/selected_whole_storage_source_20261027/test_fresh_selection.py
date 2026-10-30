"""Finite artificial fixtures only. No ML, datasets, real weights or timing.

Only this focused suite is authorized by the local source-candidate task. The
fixture review/GO words are deliberately artificial, never execution authority.
"""
import ast
import copy
import math
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import numpy as np
import prepare_fresh_selection as a

HERE = Path(__file__).resolve().parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(a.encoded(value))


def zeros(shapes, value):
    return {k: np.full(shape, value, dtype='<f4') for k, shape in shapes.items()}


class Fixture:
    """Producer-shaped metadata with newly generated, labelled artificial arrays."""
    def __init__(self, base):
        self.base = base
        self.p, self.g, self.gpkg = a.dependencies()
        self.codec = a.load_module(HERE / 'storage_codec.py', '_fixture_codec')
        self.mw, self.gw = base / 'matched', base / 'grid'
        self.mid, self.gid = 'replication-artificial-matched', 'replication-artificial-grid'
        self.mroot = self.mw / 'artifacts' / self.mid
        self.mc = self.mw / 'custody/matched' / self.mid
        self.groot = self.gw / 'artifacts' / self.gid
        self.gc = self.gw / 'custody/grid' / self.gid
        self.descriptor, self.review = base / 'parent.json', base / 'input-review.json'
        self.output = base / 'output'
        for root, paths in ((self.mroot, self.p.ARTIFACT_PATHS), (self.mc, self.p.CUSTODY_PATHS),
                            (self.groot, a.ARTIFACT_PATHS), (self.gc, a.CUSTODY_PATHS)):
            for rel in paths:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'{}\n' if rel.endswith('.json') else b'ARTIFICIAL opaque fixture; no real model/data\n')
        matched, dep = self.p.dependency(self.gpkg)
        self.dep = dep
        self.mrows, self.grows, self.attempts = [], [], []
        self.choices = {i: {'C': a.BUDGETS[i % 7], 'J': a.BUDGETS[(i + 3) % 7]} for i in range(16)}
        for seed in a.read(matched / 'SEEDS.json')['seeds']:
            i = seed['id']
            self.attempts.append({'id': i, 'returncode': 0, 'reason': None, 'elapsed_seconds': 1})
            teacher = zeros(self.codec.shapes_teacher(), (i + 1) / 100)
            np.savez_compressed(self.mroot / f'seed_{i}/teacher_final.npz', **teacher)
            logical = self.codec.state_hash(teacher)
            row = {'id': i, 'seed': seed, 'status': 'COMPLETE', 'teacher_correct': 265,
                   'teacher_initial_sha256': logical, 'teacher_final_sha256': logical, 'schedule_sha256': '3' * 64,
                   'arms': {}, 'fit_records': {}}
            for arm in ('L', 'J', 'C'):
                arrays = zeros(self.codec.shapes_arm('J' if arm == 'L' else arm, 2048), (i + 1) / 100)
                np.savez_compressed(self.mroot / f'seed_{i}/{arm}_final.npz', **arrays)
                passed = True if arm == 'L' else self.flag(i, arm, 2048)
                row['arms'][arm] = self.cell_values(passed, self.codec.state_hash(arrays))
                row['fit_records'][arm] = {'schedule_sha256': '3' * 64, 'parameter_steps': [600] * (2 if arm == 'C' else 4), 'examples_per_parameter': 76800}
            self.mrows.append(row)
            write(self.mroot / f'seed_{i}/result.json', row)
            write(self.mroot / f'seed_{i}_EXIT.json', self.attempts[-1])
            write(self.mc / f'seed_{i}.started.json', {'id': i, 'utc': '2026-01-01T00:00:00Z'})
        run = {'contract': self.p.MATCHED_CONTRACT, 'replication_id': self.mid, 'evidence_class': 'new-replication-on-seen-digits',
               'historical_attempt': self.p.CLOSED, 'package_sha256': dep['matched_package_sha256'],
               'files': dep['matched_required_files'], 'deadline_utc': '2026-01-01T01:00:00Z'}
        write(self.mroot / 'run_contract.json', run)
        write(self.mroot / 'SUMMARY.json', {'decision': 'NOT_JOINTLY_SUPPORTED', 'all_rows': self.mrows,
              'attempts': self.attempts, 'not_started': [], 'abort_reason': None, 'finished_utc': '2026-01-01T00:02:00Z'})
        self.custody(self.mc, self.mroot, run)
        self.refresh_matched()
        for seed in a.read(self.gpkg / 'SEEDS.json')['seeds']:
            i = seed['id']
            old = self.mrows[i]
            row = {'id': i, 'status': 'COMPLETE', 'parent_result_sha256': a.sha(self.mroot / f'seed_{i}/result.json'),
                   'parent_teacher_initial_sha256': old['teacher_initial_sha256'], 'parent_teacher_final_sha256': old['teacher_final_sha256'],
                   'schedule_sha256': old['schedule_sha256'], 'teacher_correct': 265,
                   'cells': [dict(old['arms'][arm], arm=arm, budget=2048, reused=True) for arm in ('J', 'C')], 'fit_records': {}}
            write(self.groot / f'seed_{i}/START.json', {'id': i, 'seed': seed, 'utc': '2026-01-01T00:03:00Z',
                  'files': self.g.manifest_hashes(self.gpkg), 'package_sha256': self.g.package_identity(self.gpkg)})
            for arm in a.ARMS:
                for budget in a.NEW_BUDGETS:
                    key = f'{arm}_{budget}'
                    arrays = zeros(self.codec.shapes_arm(arm, budget), (i + 2) / 100)
                    np.savez_compressed(self.groot / f'seed_{i}/{key}_final.npz', **arrays)
                    logical = self.codec.state_hash(arrays)
                    row['cells'].append(dict(self.cell_values(self.flag(i, arm, budget), logical), arm=arm, budget=budget, reused=False))
                    row['fit_records'][key] = {'schedule_sha256': '3' * 64, 'parameter_steps': [600] * (2 if arm == 'C' else 4), 'examples_per_parameter': 76800}
                    write(self.groot / f'seed_{i}/{key}_START.json', {'arm': arm, 'budget': budget, 'utc': '2026-01-01T00:03:00Z',
                          'schedule_sha256': '3' * 64, 'initial_state_sha256': logical})
            self.grows.append(row)
            write(self.groot / f'seed_{i}/result.json', row)
            write(self.groot / f'seed_{i}_EXIT.json', self.attempts[i])
            write(self.gc / f'seed_{i}.started.json', {'id': i, 'utc': '2026-01-01T00:03:00Z'})
        run = {'contract': self.g.CONTRACT, 'replication_id': self.gid, 'evidence_class': self.g.EVIDENCE_CLASS,
               'historical_attempt': self.g.HISTORICAL_ATTEMPT, 'package_sha256': self.g.package_identity(self.gpkg),
               'files': self.g.manifest_hashes(self.gpkg), 'deadline_utc': '2026-01-01T01:00:00Z',
               'parent_manifest_sha256': a.sha(self.descriptor), 'parent_replication_id': self.mid, 'matched_package_sha256': dep['matched_package_sha256']}
        write(self.groot / 'run_contract.json', run)
        self.custody(self.gc, self.groot, run, '2026-01-01T00:03:00Z')
        self.refresh_grid()

    def flag(self, i, arm, budget):
        minimum = self.choices[i][arm]
        return budget == minimum or (budget == 6144 and minimum < 4096)

    @staticmethod
    def cell_values(passed, logical):
        return {'nmse': .01 if passed else .2, 'correct': 265, 'n': 270, 'accuracy': 265 / 270,
                'quality_pass': bool(passed), 'state_sha256': logical}

    def custody(self, registry, root, run, start='2026-01-01T00:00:00Z'):
        release = dict(run, custody_attestation='SINGLE_DURABLE_ROOT_NO_PRIOR_ATTEMPT_NO_RETRY', parent_approved=True,
                       independent_review='GO', reviewer='ARTIFICIAL FIXTURE ONLY', approval_evidence='NOT EXECUTION AUTHORITY')
        write(registry / 'cohort.json', {'status': 'STARTED', 'utc': start, 'out': str(root), 'token_hash': '5' * 64,
              'release': release, 'end_timestamp': self.p.timestamp('2026-01-01T01:00:00Z')})
        write(registry / 'claim.json', {'replication_id': run['replication_id'], 'package_sha256': run['package_sha256'],
              'output_relative': 'artifacts/' + run['replication_id']})

    def manifests(self, root, registry, paths, seed_files):
        for i in range(16):
            write(root / f'seed_{i}/FILES_SHA256.json', {n: a.sha(root / f'seed_{i}' / n) for n in seed_files if n != 'FILES_SHA256.json'})
        write(root / 'FILES_SHA256.json', {n: a.sha(root / n) for n in paths if n != 'FILES_SHA256.json'})
        summary = a.read(root / 'SUMMARY.json')
        write(registry / 'FINISHED.json', {'utc': summary['finished_utc'], 'decision': summary['decision'], 'summary_sha256': a.sha(root / 'SUMMARY.json')})

    def refresh_matched(self):
        self.manifests(self.mroot, self.mc, self.p.ARTIFACT_PATHS, self.p.SEED_FILES)
        write(self.descriptor, self.p.validate_completed_parent(self.mw, self.mid, self.gpkg))

    def refresh_grid(self, summaries=True):
        if summaries:
            rows = [a.read(self.groot / f'seed_{i}/result.json') for i in range(16)]
            minima, patterns, counts, coverage = a.selection_metadata(rows)
            write(self.groot / 'SUMMARY.json', {'decision': 'UNCERTAIN' if coverage['both'] == 16 else 'CENSORED_NO_COHORT_RATIO',
                  'all_rows': rows, 'minima': minima, 'patterns': patterns, 'pass_counts': counts, 'coverage': coverage,
                  'attempts': self.attempts, 'not_started': [], 'abort_reason': None, 'finished_utc': '2026-01-01T00:05:00Z',
                  'scope': 'ARTIFICIAL METADATA; no science or bootstrap executed'})
        self.manifests(self.groot, self.gc, a.ARTIFACT_PATHS, a.SEED_FILES)
        self.refresh_review()

    def refresh_review(self):
        write(self.review, {'schema': a.REVIEW_SCHEMA, 'purpose': 'fresh-grid-to-storage-only',
              'matched_descriptor_sha256': a.sha(self.descriptor), 'grid_replication_id': self.gid,
              'grid_files': self.p.complete_files(self.groot, a.ARTIFACT_PATHS),
              'grid_custody_files': self.p.complete_files(self.gc, a.CUSTODY_PATHS), 'adapter_files': a.adapter_files(),
              'upstream_provenance_approved': True, 'independent_review': 'GO', 'reviewer': 'ARTIFICIAL FIXTURE ONLY',
              'approval_evidence': 'NOT EXECUTION AUTHORITY'})

    def run(self, output=None):
        return a.prepare(self.mw, self.descriptor, self.gw, self.gid, self.review, a.sha(self.review), output or self.output)


class FreshSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='canaria-artificial-selection-')
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(Path(self.temp.name))

    def reject(self, regex='.'):
        with self.assertRaisesRegex((ValueError, FileNotFoundError), regex):
            self.f.run()
        self.assertFalse(self.f.output.exists())

    def mutate_row(self, action):
        path = self.f.groot / 'seed_0/result.json'
        row = a.read(path)
        action(row)
        write(path, row)
        try:
            self.f.refresh_grid()
        except KeyError:
            summary = a.read(self.f.groot / 'SUMMARY.json')
            summary['all_rows'] = [a.read(self.f.groot / f'seed_{i}/result.json') for i in range(16)]
            write(self.f.groot / 'SUMMARY.json', summary)
            self.f.refresh_grid(False)

    def test_exact_deterministic_48_sources_all_budgets_and_pending_gate(self):
        gate = self.f.run()
        self.assertEqual(gate['status'], 'PENDING_CUSTODIAN_REVIEW')
        self.assertIs(gate['storage_execution_authorized'], False)
        origin = a.read(self.f.output / 'FRESH_SELECTION.json')
        manifest = a.read(self.f.output / 'selected_models.json')
        self.assertEqual(len(list((self.f.output / 'inputs').glob('*.npz'))), 48)
        self.assertEqual(origin['cell_count'], 224)
        self.assertTrue(any(p['nonmonotone'] for p in origin['patterns']))
        self.assertEqual(manifest['selection_origin_sha256'], a.sha(self.f.output / 'FRESH_SELECTION.json'))
        seen_budgets, stages = set(), set()
        for entry in origin['source_map']:
            source = self.f.mroot if entry['source_stage'] == 'matched' else self.f.groot
            self.assertEqual((source / entry['source_path']).read_bytes(), (self.f.output / entry['destination_path']).read_bytes())
            if entry['arm'] != 'teacher':
                seen_budgets.add(entry['budget'])
                self.assertEqual(entry['budget'], self.f.choices[entry['id']][entry['arm']])
                self.assertEqual(entry['source_stage'], 'matched' if entry['budget'] == 2048 else 'grid')
            stages.add(entry['source_stage'])
        self.assertEqual(seen_budgets, set(a.BUDGETS))
        self.assertEqual(stages, {'matched', 'grid'})
        second = self.f.base / 'second-output'
        self.f.run(second)
        self.assertEqual({p.relative_to(self.f.output): p.read_bytes() for p in self.f.output.rglob('*') if p.is_file()},
                         {p.relative_to(second): p.read_bytes() for p in second.rglob('*') if p.is_file()})
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('sklearn', sys.modules)

    def test_summary_only_is_not_accepted(self):
        (self.f.groot / 'seed_0/C_512_initial.npz').unlink()
        self.reject('census')

    def test_extra_file_is_not_accepted(self):
        (self.f.groot / 'extra').write_text('artificial')
        self.reject('census')

    def test_corrupt_unselected_opaque_bytes(self):
        (self.f.groot / 'seed_0/C_1024_initial.npz').write_bytes(b'changed artificial')
        self.reject('root manifest')

    def test_nested_manifest_not_summary_trust(self):
        write(self.f.groot / 'seed_0/FILES_SHA256.json', {})
        write(self.f.groot / 'FILES_SHA256.json', {n: a.sha(self.f.groot / n) for n in a.ARTIFACT_PATHS if n != 'FILES_SHA256.json'})
        self.f.refresh_review()
        self.reject('nested manifest')

    def test_missing_or_false_upstream_review(self):
        value = a.read(self.f.review)
        value['upstream_provenance_approved'] = False
        write(self.f.review, value)
        self.reject('reviewed upstream')

    def test_review_full_bytes_not_recomputed_trust(self):
        value = a.read(self.f.review)
        value['grid_files']['seed_0.log'] = 'a' * 64
        write(self.f.review, value)
        self.reject('full-byte')

    def test_missing_upstream_release_approval(self):
        value = a.read(self.f.gc / 'cohort.json')
        value['release']['parent_approved'] = False
        write(self.f.gc / 'cohort.json', value)
        self.f.refresh_review()
        self.reject('approval')

    def test_late_grid_completion_even_rehashed(self):
        value = a.read(self.f.groot / 'SUMMARY.json')
        value['finished_utc'] = '2026-01-01T01:00:01Z'
        write(self.f.groot / 'SUMMARY.json', value)
        self.f.refresh_grid(False)
        self.reject('deadline')

    def test_failed_exit_not_dropped(self):
        value = a.read(self.f.groot / 'SUMMARY.json')
        value['attempts'][0]['returncode'] = 1
        write(self.f.groot / 'SUMMARY.json', value)
        self.f.refresh_grid(False)
        self.reject('successful exit')

    def test_missing_cell(self):
        self.mutate_row(lambda r: r['cells'].pop())
        self.reject('224-cell')

    def test_duplicate_cell(self):
        self.mutate_row(lambda r: r['cells'].append(copy.deepcopy(r['cells'][0])))
        self.reject('224-cell')

    def test_counterfeit_pass_flag(self):
        self.mutate_row(lambda r: r['cells'][0].update(quality_pass=not r['cells'][0]['quality_pass']))
        self.reject('pass/accuracy')

    def test_later_passing_budget_substitution(self):
        value = a.read(self.f.groot / 'SUMMARY.json')
        value['minima'][0]['C'] = 6144
        write(self.f.groot / 'SUMMARY.json', value)
        self.f.refresh_grid(False)
        self.reject('minima')

    def test_reused_2048_cell_state_changed(self):
        self.mutate_row(lambda r: r['cells'][0].update(state_sha256='f' * 64))
        self.reject('reused 2048')

    def test_teacher_source_state_mismatch(self):
        path = self.f.mroot / 'seed_0/teacher_final.npz'
        np.savez_compressed(path, **zeros(self.f.codec.shapes_teacher(), .999))
        self.f.refresh_matched()
        for root in (self.f.groot,):
            run = a.read(root / 'run_contract.json')
            run['parent_manifest_sha256'] = a.sha(self.f.descriptor)
            write(root / 'run_contract.json', run)
            cohort = a.read(self.f.gc / 'cohort.json')
            cohort['release']['parent_manifest_sha256'] = a.sha(self.f.descriptor)
            write(self.f.gc / 'cohort.json', cohort)
        self.f.refresh_grid()
        self.reject('logical state')

    def test_corrupt_selected_npz_with_refreshed_file_pins(self):
        (self.f.groot / 'seed_0/C_512_final.npz').write_bytes(b'ARTIFICIAL corrupt NPZ')
        self.f.refresh_grid()
        with self.assertRaises(zipfile.BadZipFile):
            self.f.run()
        self.assertFalse(self.f.output.exists())

    def test_censoring_preserved_no_output(self):
        # Choose a seed where J_2048 is already a failing reused cell.
        path = self.f.groot / 'seed_1/result.json'
        row = a.read(path)
        for cell in row['cells']:
            if cell['arm'] == 'J':
                cell.update(nmse=.2, quality_pass=False)
        write(path, row)
        self.f.refresh_grid()
        self.reject('censored cohort')

    def test_grid_parent_identity_wrong(self):
        self.mutate_row(lambda r: r.update(parent_teacher_final_sha256='e' * 64))
        self.reject('parent binding')

    def test_boolean_id_rejected(self):
        self.mutate_row(lambda r: r.update(id=False))
        self.reject('integer teacher')

    def asymmetric(self, path, mutate, regex):
        """Change only one cross-bound JSON copy; rehash, never repair its peer."""
        original = a.read(path)
        changed = copy.deepcopy(original)
        mutate(changed)
        write(path, changed)
        self.f.refresh_grid(False)
        self.reject(regex)
        write(path, original)
        self.f.refresh_grid(False)

    def test_asymmetric_result_id_types(self):
        for i, value in ((0, False), (0, 0.0), (1, True), (1, 1.0)):
            with self.subTest(id=i, value=repr(value)):
                self.asymmetric(self.f.groot / f'seed_{i}/result.json',
                                lambda row: row.update(id=value), 'integer result')

    def test_asymmetric_exit_id_and_returncode_types(self):
        for i, value in ((0, False), (0, 0.0), (1, True), (1, 1.0)):
            with self.subTest(id=i, value=repr(value)):
                self.asymmetric(self.f.groot / f'seed_{i}_EXIT.json',
                                lambda row: row.update(id=value), 'integer exit')
        for value in (False, 0.0):
            with self.subTest(returncode=repr(value)):
                self.asymmetric(self.f.groot / 'seed_0_EXIT.json',
                                lambda row: row.update(returncode=value), 'exit/result mismatch')

    def test_asymmetric_custody_admission_id_types(self):
        for i, value in ((0, False), (0, 0.0), (1, True), (1, 1.0)):
            with self.subTest(id=i, value=repr(value)):
                self.asymmetric(self.f.gc / f'seed_{i}.started.json',
                                lambda row: row.update(id=value), 'integer admission')

    def test_asymmetric_worker_start_id_types(self):
        for i, value in ((0, False), (0, 0.0), (1, True), (1, 1.0)):
            with self.subTest(id=i, value=repr(value)):
                self.asymmetric(self.f.groot / f'seed_{i}/START.json',
                                lambda row: row.update(id=value), 'integer START')
        self.asymmetric(self.f.groot / 'seed_0/START.json',
                        lambda row: row['seed'].update(id=False), 'seed source binding')

    def test_asymmetric_summary_cell_types(self):
        for field, value in (('quality_pass', 1), ('reused', 1), ('budget', 2048.0), ('n', 270.0)):
            with self.subTest(field=field):
                self.asymmetric(self.f.groot / 'SUMMARY.json',
                                lambda summary: summary['all_rows'][0]['cells'][0].update({field: value}),
                                'summary/result mismatch')

    def test_asymmetric_result_cell_types(self):
        for field, value in (('quality_pass', 1), ('reused', 1), ('budget', 2048.0), ('n', 270.0)):
            with self.subTest(field=field):
                self.asymmetric(self.f.groot / 'seed_0/result.json',
                                lambda row: row['cells'][0].update({field: value}), 'summary/result mismatch')

    def test_duplicate_json_field(self):
        path = self.f.groot / 'SUMMARY.json'
        path.write_bytes(b'{"decision":"UNCERTAIN",' + path.read_bytes()[1:])
        self.reject('duplicate JSON')

    def test_nonfinite_json(self):
        (self.f.groot / 'seed_0/environment.json').write_text('{"x":NaN}')
        self.reject('nonfinite JSON')

    def test_symlink_input(self):
        path = self.f.groot / 'seed_0/C_512_final.npz'
        path.unlink()
        path.symlink_to(self.f.groot / 'seed_1/C_512_final.npz')
        self.reject('symlink')

    def test_existing_output_rejected(self):
        self.f.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh and absent'):
            self.f.run()
        self.assertEqual(list(self.f.output.iterdir()), [])

    def test_output_overlap(self):
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            self.f.run(self.f.gw / 'new-output')
        self.assertFalse((self.f.gw / 'new-output').exists())

    def test_overflow_json_rejected(self):
        (self.f.groot / 'seed_0/environment.json').write_text('{"x":1e999}')
        self.reject('nonfinite JSON')

    def test_producer_selection_prefix_agrees_without_bootstrap(self):
        # AST-extract only the original pure selection prefix; do not import a
        # runner or execute its bootstrap branch, workers, data or ML imports.
        tree = ast.parse((self.f.gpkg / 'runner.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'analyze')
        branch = next(i for i, n in enumerate(function.body)
                      if isinstance(n, ast.If) and ast.unparse(n.test) == "coverage['both'] == 16")
        function.body = function.body[:branch] + [ast.Return(value=ast.Name(id='out', ctx=ast.Load()))]
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        scope = {'math': math, 'BUDGETS': a.BUDGETS}
        exec(compile(module, '<reviewed-grid-selection-prefix>', 'exec'), scope)
        for censor in (False, True):
            rows = copy.deepcopy(self.f.grows)
            if censor:
                for cell in rows[1]['cells']:
                    if cell['arm'] == 'J':
                        cell.update(nmse=.2, quality_pass=False)
            original = scope['analyze'](rows)
            observed = a.selection_metadata(rows)
            self.assertEqual(observed, tuple(original[k] for k in ('minima', 'patterns', 'pass_counts', 'coverage')))

    def test_source_binding_and_path_census(self):
        self.assertEqual((len(a.ARTIFACT_PATHS), len(a.CUSTODY_PATHS), len(a.SEED_FILES)), (883, 19, 53))
        value = a.read(self.f.review)
        value['adapter_files']['prepare_fresh_selection.py'] = '0' * 64
        write(self.f.review, value)
        self.reject('source binding')


if __name__ == '__main__':
    unittest.main(verbosity=2)
