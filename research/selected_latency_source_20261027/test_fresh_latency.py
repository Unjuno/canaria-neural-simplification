"""Finite ARTIFICIAL fixtures only, never actual scientific validation or timing.

One narrow CSV-digest mock enables a positive metadata route without opening the
frozen dataset. An unmocked negative check proves the artificial file is rejected.
No historical constant changes; no torch, sklearn, runner, network or attempt.
"""
import ast
import contextlib
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
import fresh_latency_contract as c
import prepare_fresh_latency as prep
import latency_contract as historical
import benchmark as b
import supervise
import timing_core as timing
from test_controls import Controls as HistoricalArtificialControls

HERE = Path(__file__).resolve().parent
STORAGE = HERE.parent / 'selected_whole_storage_source_20261027'
sys.path.insert(0, str(STORAGE))
import test_fresh_selection as selection_fixtures
import prepare_fresh_selection as selection_adapter
import measure as storage_measure


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(c.encoded(value))


class Fixture:
    def __init__(self, root):
        self.root = root
        self.f = selection_fixtures.Fixture(root)
        self.csv = root / 'artificial.csv.gz'
        self.csv.write_bytes(b'ARTIFICIAL ONLY; not digits CSV; never used for inference\n')
        # These references are newly generated fixtures, not actual saved logits.
        for i in range(16):
            np.savez_compressed(self.f.mroot / f'seed_{i}/split_ids.npz', train=np.arange(1077), validation=np.arange(270))
            np.savez_compressed(self.f.mroot / f'seed_{i}/validation_arrays.npz',
                **{k: np.zeros((270, 10), dtype='<f4') for k in ('teacher_logits', 'C_logits', 'J_logits')})
            np.savez_compressed(self.f.groot / f'seed_{i}/validation_arrays.npz',
                **{f'{a}_{budget}_logits': np.zeros((270, 10), dtype='<f4') for a in ('C', 'J') for budget in c.historical.storage.BUDGETS if budget != 2048})
        self.f.refresh_matched()
        run = c.read_json(self.f.groot / 'run_contract.json')
        run['parent_manifest_sha256'] = c.sha(self.f.descriptor)
        write(self.f.groot / 'run_contract.json', run)
        self.f.custody(self.f.gc, self.f.groot, run, '2026-01-01T00:03:00Z')
        self.f.refresh_grid()
        self.f.run()
        self.stored = root / 'storage'
        manifest = c.read_json(self.f.output / 'selected_models.json')
        result = storage_measure.measure(manifest, self.f.output, self.stored, c.sha(self.f.output / 'selected_models.json'))
        assert result['status'] == 'COMPLETE'
        self.review = root / 'fresh-review.json'
        self.kwargs = dict(input_root=root, selection_directory='output', storage_directory='storage',
                          matched_directory=self.f.mroot.relative_to(root).as_posix(),
                          grid_directory=self.f.groot.relative_to(root).as_posix(), parent_descriptor='parent.json',
                          upstream_review='input-review.json', csv='artificial.csv.gz')
        self.original_sha = c.sha
        self.manifest_path = root / 'artificial-latency-input.json'
        with self.fake_csv():
            self.refresh()

    def fake_csv(self):
        def digest(path):
            if Path(path).resolve() == self.csv.resolve():
                assert self.csv.read_bytes().startswith(b'ARTIFICIAL ONLY')
                return c.CSV_SHA256
            return self.original_sha(path)
        return mock.patch.object(c, 'sha', side_effect=digest)

    def refresh(self):
        self.manifest = prep.describe_inputs(**self.kwargs)
        review = {'schema': c.REVIEW_SCHEMA, 'purpose': 'fresh-storage-to-latency-only',
                  'input_basis_sha256': c.basis_sha256(self.manifest), 'source_manifest': c.package_hashes(HERE),
                  'selection_review': 'GO', 'storage_execution_approved': True, 'storage_complete_review': 'GO',
                  'reference_provenance_review': 'GO', 'independent_review': 'GO',
                  'reviewer': 'ARTIFICIAL FIXTURE ONLY', 'approval_evidence': 'NOT EXECUTION AUTHORITY'}
        write(self.review, review)
        self.manifest['input_review'] = {'path': 'fresh-review.json', 'sha256': c.sha(self.review)}
        write(self.manifest_path, self.manifest)

    def metadata(self):
        return c.load_metadata(self.manifest, self.root)

    def mutate(self, path, action):
        value = c.read_json(path); action(value); write(path, value)
        with self.fake_csv():
            self.refresh()


class FreshBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='canaria-artificial-fresh-latency-')
        cls.root = Path(cls.temp.name)
        cls.template = cls.root / 'template'; cls.template.mkdir()
        cls.fixture = Fixture(cls.template)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        # Reuse one deterministic artificial bundle. Restore any modified files
        # after each test; no real registry or real scientific output is involved.
        self.f = self.fixture
        self.saved = {p: p.read_bytes() for p in self.f.root.rglob('*') if p.is_file()}
        self.addCleanup(self.restore)
        self.patch = self.f.fake_csv(); self.patch.start(); self.addCleanup(self.patch.stop)
        self.f.refresh()

    def restore(self):
        for p in self.f.root.rglob('*'):
            if p.is_file() and p not in self.saved:
                p.unlink()
        for p, raw in self.saved.items():
            if p.is_symlink():
                p.unlink()
            p.write_bytes(raw)
        self.f.manifest = c.read_json(self.f.manifest_path)

    def reject(self, regex='.'):
        with self.assertRaisesRegex((ValueError, FileNotFoundError), regex):
            self.f.metadata()

    def test_complete_artificial_bundle_all_budgets_and_exact_48_states(self):
        inputs = c.load_inputs(self.f.manifest_path, c.sha(self.f.manifest_path), self.f.root)
        self.assertEqual(len(inputs['models']), 16)
        self.assertEqual(len(inputs['source_map']), 48)
        self.assertEqual(len(inputs['artifacts']), 96)
        budgets = set()
        for item in inputs['models']:
            for arm in ('C', 'J'):
                budget = item['selection']['arms'][arm]['budget']; budgets.add(budget)
                expected = arm + '_logits' if budget == 2048 else f'{arm}_{budget}_logits'
                self.assertEqual(item['validation']['logits'][arm]['key'], expected)
                path = item['references'][arm]
                self.assertTrue(path.is_relative_to(self.f.f.mroot if budget == 2048 else self.f.f.groot))
        self.assertEqual(budgets, set(historical.storage.BUDGETS))
        self.assertNotIn('torch', sys.modules); self.assertNotIn('sklearn', sys.modules)

    def test_original_frozen_csv_gate_rejects_artificial_bytes_unmocked(self):
        self.patch.stop()
        self.assertNotEqual(c.sha(self.f.csv), c.CSV_SHA256)
        self.reject('SHA256')
        self.patch.start()

    def test_only_48_stored_archives_required_not_deflated_payloads(self):
        for path in self.f.stored.glob('*_deflated.npz'):
            path.unlink()
        self.assertEqual(len(self.f.metadata()['models']), 16)

    def test_missing_reference_is_not_generated(self):
        path = self.f.f.mroot / 'seed_0/validation_arrays.npz'; path.unlink()
        self.reject()
        self.assertFalse(path.exists())

    def test_saved_reference_byte_drift(self):
        (self.f.f.groot / 'seed_0/validation_arrays.npz').write_bytes(b'ARTIFICIAL changed')
        self.reject('SHA256')

    def test_reference_key_or_2048_route_cannot_be_relabelled(self):
        for field, value in (('key', 'teacher_logits'), ('sha256', 'a' * 64), ('path', 'arbitrary.npz')):
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.f.manifest)
                manifest['validation'][0]['logits']['C'][field] = value
                # Updating only a purported review never turns this into genuine upstream reference provenance.
                review = c.read_json(self.f.review); review['input_basis_sha256'] = c.basis_sha256(manifest)
                write(self.f.review, review); manifest['input_review']['sha256'] = c.sha(self.f.review)
                with self.assertRaisesRegex(ValueError, 'saved validation'):
                    c.load_metadata(manifest, self.f.root)

    def test_pending_only_is_never_review_approval(self):
        self.f.manifest['input_review'] = copy.deepcopy(self.f.manifest['pending_gate'])
        self.reject('review')

    def test_review_false_missing_or_stale_fails_closed(self):
        original = c.read_json(self.f.review)
        for field, value in (('selection_review', 'PENDING'), ('storage_complete_review', 'NO'),
                             ('reference_provenance_review', 'NO'), ('storage_execution_approved', False),
                             ('independent_review', 'PENDING'), ('reviewer', ''), ('input_basis_sha256', 'a' * 64),
                             ('source_manifest', {})):
            with self.subTest(field=field):
                changed = dict(original); changed[field] = value; write(self.f.review, changed)
                self.f.manifest['input_review']['sha256'] = c.sha(self.f.review)
                self.reject()
        self.f.review.unlink(); self.reject()

    def test_original_pending_gate_cannot_be_promoted(self):
        self.f.mutate(self.f.f.output / 'REVIEW_REQUIRED.json', lambda x: x.update(storage_execution_authorized=True))
        self.reject('immutable selection gate')

    def test_manifest_origin_parent_summary_and_upstream_review_bindings(self):
        for key in ('fresh_selection', 'parent_descriptor', 'upstream_review', 'grid_summary'):
            with self.subTest(key=key):
                manifest = copy.deepcopy(self.f.manifest); manifest[key]['sha256'] = 'a' * 64
                review = c.read_json(self.f.review); review['input_basis_sha256'] = c.basis_sha256(manifest)
                write(self.f.review, review); manifest['input_review']['sha256'] = c.sha(self.f.review)
                with self.assertRaises(ValueError):
                    c.load_metadata(manifest, self.f.root)

    def test_selection_budget_drift_rejected(self):
        self.f.mutate(self.f.f.output / 'selected_models.json', lambda x: x['models'][0]['arms']['C'].update(budget=1024))
        self.reject('selection gate')

    def test_grid_cell_boolean_identity_and_censoring_rejected(self):
        inputs = self.f.metadata(); origin = c.read_json(self.f.f.output / 'FRESH_SELECTION.json')
        selected = c.read_json(self.f.f.output / 'selected_models.json')
        parent = c.read_json(self.f.f.descriptor); upstream = c.read_json(self.f.f.review)
        a, p, g, grid, _ = c.dependencies()
        for change in (lambda x: x['all_rows'][0].update(id=False),
                       lambda x: x['all_rows'][0]['cells'][0].update(budget=2048.0),
                       lambda x: x.update(decision='CENSORED_NO_COHORT_RATIO'),
                       lambda x: x['minima'][0].update(C=6144),
                       lambda x: x['all_rows'][0]['cells'].append(x['all_rows'][0]['cells'][0])):
            bad = copy.deepcopy(origin); change(bad['grid_summary'])
            with self.assertRaises(ValueError):
                c.check_origin(bad, selected, parent, upstream, bad['grid_summary'], a, p, g, grid)

    def test_source_hash_and_state_metadata_drift(self):
        path = self.f.f.output / 'inputs/seed_0_teacher.npz'; path.write_bytes(b'ARTIFICIAL corrupt source')
        self.reject('SHA256')

    def test_storage_incomplete_and_duplicate_boolean_census(self):
        path = self.f.stored / 'CENSUS.json'
        self.f.mutate(path, lambda x: x[0].update(id=False))
        self.reject('census')

    def test_storage_incomplete_summary(self):
        self.f.mutate(self.f.stored / 'SUMMARY.json', lambda x: x.update(status='INCOMPLETE'))
        self.reject('incomplete')

    def test_storage_source_and_selection_environment_binding(self):
        self.f.mutate(self.f.stored / 'ENV.json', lambda x: x.update(manifest_sha256='a' * 64))
        self.reject('selection-manifest')

    def test_storage_source_hash_mapping_is_not_ignored(self):
        self.f.mutate(self.f.stored / 'SOURCES.json', lambda x: x[1].update(logical_sha256='a' * 64))
        self.reject('source/state')

    def test_stored_archive_byte_drift(self):
        (self.f.stored / 'seed_0_teacher_stored.npz').write_bytes(b'ARTIFICIAL corrupt archive')
        self.reject('archive byte')

    def test_stored_content_mismatch_even_when_new_hash_is_supplied(self):
        inputs = self.f.metadata(); codec = self.f.f.codec
        name = 'seed_0_teacher_stored.npz'; path = self.f.stored / name
        arrays = codec.load_source(path, inputs['artifacts'][name]['sha256'], codec.shapes_teacher())
        arrays['stem.bias'][0] += np.float32(1)
        path.unlink(); codec.serialize(path, arrays, codec.zipfile.ZIP_STORED)
        inputs['artifacts'][name]['sha256'] = c.sha(path)
        with self.assertRaisesRegex(ValueError, 'content mismatch'):
            c.verify_payloads(inputs)

    def test_teacher_and_arm_logical_states_are_checked(self):
        for arm in ('teacher', 'C'):
            with self.subTest(arm=arm):
                inputs = self.f.metadata()
                if arm == 'teacher': inputs['source_map'][0, arm]['state_sha256'] = 'a' * 64
                else: inputs['models'][0]['selection']['arms'][arm]['state_sha256'] = 'a' * 64
                with self.assertRaisesRegex(ValueError, 'logical state'):
                    c.verify_payloads(inputs)

    def test_tensor_hash_and_parameter_counts_checked(self):
        inputs = self.f.metadata(); inputs['tensors'][0]['sha256'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'tensor-content'):
            c.verify_payloads(inputs)
        inputs = self.f.metadata(); inputs['models'][0]['expected']['teacher_parameters'] += 1
        with self.assertRaisesRegex(ValueError, 'parameter/payload'):
            c.verify_payloads(inputs)

    def test_missing_nonfinite_logit_or_invalid_split_fails(self):
        for kind in ('missing', 'nonfinite', 'split'):
            with self.subTest(kind=kind):
                inputs = copy.deepcopy(self.f.metadata()); item = inputs['models'][0]
                if kind == 'split':
                    path = item['split']; np.savez_compressed(path, validation=np.zeros(270, dtype=np.int64))
                    item['validation']['split']['sha256'] = c.sha(path)
                else:
                    path = item['references']['teacher']
                    values = np.zeros((270, 10), dtype='<f4')
                    if kind == 'nonfinite': values[0, 0] = np.nan
                    np.savez_compressed(path, **{('not_a_reference' if kind == 'missing' else 'teacher_logits'): values})
                    item['validation']['logits']['teacher']['sha256'] = c.sha(path)
                with self.assertRaises(ValueError):
                    c.verify_payloads(inputs)
                for p in (item['split'], item['references']['teacher']):
                    p.write_bytes(self.saved[p])

    def test_strict_json_path_and_identity(self):
        for change in (lambda m: m['csv'].update(path='../outside.gz'),
                       lambda m: m['validation'][0].update(id=False),
                       lambda m: m['validation'][0].update(id=0.0),
                       lambda m: m['storage']['files'].pop('TENSORS.json')):
            bad = copy.deepcopy(self.f.manifest); change(bad)
            with self.assertRaises(ValueError): c.validate_inputs(bad)
        path = self.f.root / 'bad.json'
        for raw in ('{"x":1,"x":2}', '{"x":1e999}', '{"x":NaN}'):
            path.write_text(raw)
            with self.assertRaises(ValueError): c.read_json(path)

    def test_symlink_is_rejected(self):
        path = self.f.f.mroot / 'seed_0/validation_arrays.npz'
        backup = path.with_name('artificial_backup.npz'); backup.write_bytes(path.read_bytes())
        path.unlink(); path.symlink_to(backup)
        self.reject('symlink')

    def test_preparation_only_writes_manifest_and_never_overwrites(self):
        target = self.root / 'prepared-private.json'
        if target.exists(): target.unlink()
        result = prep.prepare(**self.f.kwargs, input_review='fresh-review.json',
                              input_review_sha256=c.sha(self.f.review), output=target)
        self.assertEqual(result['status'], 'PREPARED_PRIVATE_INPUT_NOT_EXECUTION_AUTHORITY')
        self.assertEqual(c.read_json(target), self.f.manifest)
        before = target.read_bytes()
        with self.assertRaisesRegex(ValueError, 'fresh and absent'):
            prep.prepare(**self.f.kwargs, input_review='fresh-review.json', input_review_sha256=c.sha(self.f.review), output=target)
        self.assertEqual(target.read_bytes(), before); target.unlink()

    def test_preparation_rejects_inside_input_before_any_output(self):
        target = self.f.root / 'must-not-write.json'
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            prep.prepare(**self.f.kwargs, input_review='fresh-review.json', input_review_sha256=c.sha(self.f.review), output=target)
        self.assertFalse(target.exists())

    def test_fresh_cannot_be_sent_through_original_historical_gate(self):
        with self.assertRaises(ValueError): historical.validate_inputs(self.f.manifest)
        self.assertEqual(c.CSV_SHA256, historical.CSV_SHA256)
        self.assertEqual(historical.ORIGIN_SELECTION_SHA256, 'd31543ed32bef5705f92f1a80974ad63bbca3c46e12d60b62d446c29f1e8bd1e')
        self.assertEqual(historical.REFERENCE_MANIFEST_SHA256, 'd7b132c953282fd0179b4da6e1e3226f60d5bad05128df5e81027ae3a50af761')
        self.assertEqual(historical.REFERENCE_SELECTION_SHA256, '6cf4c87e8c7ca43f8760fffae8ea1155dfcccf47e61f778f2d4cc436d50d01f3')


class FreshControlTests(unittest.TestCase):
    def test_release_schema_separation_and_common_bindings(self):
        release = {'schema': c.RELEASE_SCHEMA, 'purpose': 'separate-authorized-technical-attempt-not-historical',
                   'attempt_id': 'artificial-only', 'control_root_sha256': 'a' * 64, 'deadline_utc': '2099-01-01T00:00:00Z',
                   'parent_approved': True, 'independent_review': 'GO', 'no_active_parallel_benchmark': True,
                   'input_manifest_sha256': 'b' * 64, 'source_manifest': {'fake': 'c' * 64}, 'implementation_commit': 'd' * 40}
        self.assertGreater(c.validate_release(release, 'b' * 64, {'fake': 'c' * 64}, 'd' * 40, 0), 0)
        with self.assertRaises(ValueError): historical.validate_release(release, 'b' * 64, {'fake': 'c' * 64}, 'd' * 40, 0)
        with self.assertRaises(ValueError): c.validate_release(dict(release, schema=historical.RELEASE_SCHEMA), 'b' * 64, {'fake': 'c' * 64}, 'd' * 40, 0)
        for key, val in (('parent_approved', False), ('independent_review', 'NO'), ('no_active_parallel_benchmark', False),
                         ('source_manifest', {}), ('input_manifest_sha256', 'e' * 64), ('implementation_commit', 'f' * 40),
                         ('deadline_utc', '1970-01-01T00:00:05Z')):
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.validate_release(dict(release, **{key: val}), 'b' * 64, {'fake': 'c' * 64}, 'd' * 40, 0)

    def test_explicit_fresh_worker_uses_fake_callables_exact_census(self):
        class ArtificialInput:
            def astype(self, _): return self
            def __truediv__(self, _): return self
            def __getitem__(self, index): return {'batch': index.stop}
        counts = {}; ticks = iter(range(0, 1000000, 100)); loads = []
        def load(selection, *_):
            seed = selection['id']; loads.append(seed)
            return {arm: (lambda x, s=seed, a=arm: counts.__setitem__((s, x['batch'], a), counts.get((s, x['batch'], a), 0) + 1)) for arm in timing.ARMS}
        fake_np = types.SimpleNamespace(float32='artificial', random=types.SimpleNamespace(default_rng=lambda seed:
            types.SimpleNamespace(integers=lambda *a, **k: ArtificialInput())), save=lambda buf, x, allow_pickle: buf.write(b'ARTIFICIAL'))
        runtime = types.SimpleNamespace(np=fake_np, torch=types.SimpleNamespace(no_grad=contextlib.nullcontext, from_numpy=lambda x: x),
                                        load_models=load, validation=lambda *a: {'ARTIFICIAL': True})
        inputs = {'models': [{'selection': {'id': i}, 'expected': {}, 'validation': {}, 'sources': {}, 'references': {}, 'split': 'ARTIFICIAL'} for i in range(16)],
                  'artifacts': {}, 'storage_root': 'ARTIFICIAL', 'csv': 'ARTIFICIAL'}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); reg = root / 'registry'; reg.mkdir(); out = root / 'out'
            args = types.SimpleNamespace(out=str(out), registry=str(reg), release='ARTIFICIAL', manifest='ARTIFICIAL',
                                         manifest_sha256='a' * 64, input_root='ARTIFICIAL', input_mode='fresh-v1')
            request = {'release_sha256': 'b' * 64, 'input_manifest_sha256': 'a' * 64}
            with contextlib.ExitStack() as stack:
                for module in (c, historical):
                    for name, value in [('read_json', request), ('package_hashes', {}), ('git_identity', 'c' * 40), ('validate_release', 1000), ('sha', 'b' * 64)]:
                        stack.enter_context(mock.patch.object(module, name, return_value=value))
                loaded = stack.enter_context(mock.patch.object(c, 'load_inputs', return_value=inputs))
                stack.enter_context(mock.patch.object(historical, 'load_inputs', side_effect=AssertionError('wrong historical route')))
                stack.enter_context(mock.patch.dict(b.os.environ, {'CANARIA_LATENCY_SUPERVISOR': str(b.os.getppid())}))
                stack.enter_context(mock.patch.object(b, 'configure_runtime', return_value=runtime))
                stack.enter_context(mock.patch.object(b, 'environment', return_value={'ARTIFICIAL': True}))
                stack.enter_context(mock.patch.object(b.resource, 'setrlimit'))
                stack.enter_context(mock.patch.object(b.signal, 'signal')); stack.enter_context(mock.patch.object(b.signal, 'alarm'))
                stack.enter_context(mock.patch.object(b.time, 'time', return_value=100))
                stack.enter_context(mock.patch.object(b, 'timed_block', side_effect=lambda fn, x: timing.timed_block(fn, x, clock=lambda: next(ticks))))
                b.run(args)
                self.assertEqual(loaded.call_count, 2)
            raw = [json.loads(line) for line in (out / 'RAW.jsonl').read_text().splitlines()]
            self.assertEqual(loads, list(range(16))); self.assertEqual(len(timing.validate_raw(raw)), 2880)
            self.assertEqual(len(counts), 96); self.assertTrue(all(n == 620 for n in counts.values()))
            self.assertEqual(json.loads((reg / 'EXIT.json').read_text())['status'], 'COMPLETE')

    def test_fresh_supervisor_forwards_mode_and_cleans_fake_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); inp = root / 'inputs'; control = root / 'control'; inp.mkdir(); control.mkdir()
            release = {'attempt_id': 'artificial-only', 'control_root_sha256': hashlib.sha256(str(control.resolve()).encode()).hexdigest()}
            proc = mock.Mock(pid=123); proc.poll.return_value = None
            proc.wait.side_effect = [RuntimeError('ARTIFICIAL wait failure'), -15]
            argv = ['--out', str(root / 'out'), '--release', 'ARTIFICIAL', '--manifest', 'ARTIFICIAL', '--manifest-sha256', 'a' * 64,
                    '--input-root', str(inp), '--control-root', str(control), '--input-mode', 'fresh-v1']
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(c, 'read_json', return_value=release))
                stack.enter_context(mock.patch.object(c, 'package_hashes', return_value={}))
                stack.enter_context(mock.patch.object(historical, 'git_identity', return_value='c' * 40))
                stack.enter_context(mock.patch.object(c, 'validate_release', return_value=1000))
                stack.enter_context(mock.patch.object(historical, 'sha', return_value='b' * 64))
                stack.enter_context(mock.patch.object(supervise.time, 'time', return_value=100))
                stack.enter_context(mock.patch.object(supervise.time, 'monotonic', return_value=0))
                popen = stack.enter_context(mock.patch.object(supervise.subprocess, 'Popen', return_value=proc))
                kill = stack.enter_context(mock.patch.object(supervise.os, 'killpg'))
                with self.assertRaises(RuntimeError): supervise.main(argv)
                kill.assert_called_once_with(123, supervise.signal.SIGTERM)
                self.assertEqual(popen.call_args.args[0][-2:], ['--input-mode', 'fresh-v1'])
            event = json.loads((control / 'attempt-artificial-only/SUPERVISOR_EXIT.json').read_text())
            self.assertEqual(event['status'], 'FAILED'); self.assertEqual(len(event['census']), 96)
            self.assertEqual(proc.wait.call_args_list[-1].kwargs, {'timeout': 5})
            with self.assertRaises(ValueError): historical.paths(root / 'out', inp, control, HERE, 'artificial-only')


if __name__ == '__main__':
    unittest.main()
