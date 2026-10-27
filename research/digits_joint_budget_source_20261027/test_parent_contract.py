"""Synthetic bytes/JSON only: no numerical packages, dataset reads or ML calls."""
import copy
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import parent_contract as p
import grid_contract as g

HERE = pathlib.Path(__file__).resolve().parent


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def release():
    dep = p.read_json(HERE / 'DEPENDENCY.json')
    return {
        'contract': g.CONTRACT, 'replication_id': 'replication-grid-synthetic',
        'evidence_class': g.EVIDENCE_CLASS, 'historical_attempt': g.HISTORICAL_ATTEMPT,
        'custody_attestation': g.CUSTODY_ATTESTATION, 'parent_approved': True,
        'independent_review': 'GO', 'reviewer': 'synthetic test',
        'approval_evidence': 'synthetic test; not execution authorization',
        'package_sha256': g.package_identity(HERE), 'files': g.manifest_hashes(HERE),
        'deadline_utc': '2030-01-01T00:00:00Z',
        'parent_manifest_sha256': 'a' * 64, 'parent_replication_id': 'replication-parent-synthetic',
        'matched_package_sha256': dep['matched_package_sha256'],
    }


def fixture(workspace):
    matched, dep = p.dependency(HERE)
    rid = 'replication-parent-synthetic'
    root = workspace / 'artifacts' / rid
    custody = workspace / 'custody/matched' / rid
    for rel in p.ARTIFACT_PATHS:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic opaque bytes; never an NPZ/NPY payload')
    for rel in p.CUSTODY_PATHS:
        write(custody / rel, {})
    seeds = p.read_json(matched / 'SEEDS.json')['seeds']
    rows = []
    attempts = []
    for seed in seeds:
        i = seed['id']
        rows.append({
            'id': i, 'seed': seed, 'status': 'COMPLETE', 'teacher_correct': 265,
            'teacher_initial_sha256': '1' * 64, 'teacher_final_sha256': '2' * 64,
            'schedule_sha256': '3' * 64,
            'arms': {a: {'nmse': .01, 'correct': 265, 'n': 270, 'accuracy': 265 / 270, 'quality_pass': True, 'state_sha256': '4' * 64} for a in ('L', 'J', 'C')},
            'fit_records': {a: {'schedule_sha256': '3' * 64, 'parameter_steps': [600] * (2 if a == 'C' else 4), 'examples_per_parameter': 76800} for a in ('L', 'J', 'C')},
        })
        write(root / f'seed_{i}/result.json', rows[-1])
        attempts.append({'id': i, 'returncode': 0, 'reason': None, 'elapsed_seconds': 1})
        write(root / f'seed_{i}_EXIT.json', attempts[-1])
        write(custody / f'seed_{i}.started.json', {'id': i, 'utc': '2026-01-01T00:00:00Z'})
    run = {
        'contract': p.MATCHED_CONTRACT, 'replication_id': rid,
        'evidence_class': 'new-replication-on-seen-digits', 'historical_attempt': p.CLOSED,
        'package_sha256': dep['matched_package_sha256'], 'files': dep['matched_required_files'],
        'deadline_utc': '2026-01-01T01:00:00Z',
    }
    write(root / 'run_contract.json', run)
    summary = {'decision': 'CONDITIONAL_REGIME_SUPPORTED', 'all_rows': rows, 'attempts': attempts, 'not_started': [], 'abort_reason': None, 'finished_utc': '2026-01-01T00:02:00Z'}
    write(root / 'SUMMARY.json', summary)
    private_release = dict(run, custody_attestation='SINGLE_DURABLE_ROOT_NO_PRIOR_ATTEMPT_NO_RETRY', parent_approved=True, independent_review='GO', reviewer='synthetic', approval_evidence='synthetic only')
    write(custody / 'cohort.json', {'status': 'STARTED', 'utc': '2026-01-01T00:00:00Z', 'out': str(root), 'token_hash': '5' * 64, 'release': private_release, 'end_timestamp': p.timestamp('2026-01-01T01:00:00Z')})
    write(custody / 'claim.json', {'replication_id': rid, 'package_sha256': dep['matched_package_sha256'], 'output_relative': 'artifacts/' + rid})
    refresh(root, custody)
    return rid, root, custody


def refresh(root, custody):
    for i in range(16):
        write(root / f'seed_{i}/FILES_SHA256.json', {n: p.sha(root / f'seed_{i}' / n) for n in p.SEED_FILES if n != 'FILES_SHA256.json'})
    write(root / 'FILES_SHA256.json', {n: p.sha(root / n) for n in p.ARTIFACT_PATHS if n != 'FILES_SHA256.json'})
    write(custody / 'FINISHED.json', {'utc': '2026-01-01T00:02:01Z', 'decision': p.read_json(root / 'SUMMARY.json')['decision'], 'summary_sha256': p.sha(root / 'SUMMARY.json')})


class ParentContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = pathlib.Path(self.temp.name)
        self.rid, self.root, self.custody = fixture(self.work)

    def manifest(self):
        return p.validate_completed_parent(self.work, self.rid, HERE)

    def test_full_parent_and_private_custody_census(self):
        manifest = self.manifest()
        self.assertEqual(len(manifest['files']), 339)
        self.assertEqual(len(manifest['custody_files']), 19)
        self.assertEqual(p.verify_parent(self.work, manifest, HERE), self.root)

    def test_expired_completed_parent_is_not_new_authority(self):
        self.assertEqual(self.manifest()['parent_contract'], p.MATCHED_CONTRACT)

    def test_opaque_artifact_tamper(self):
        manifest = self.manifest()
        (self.root / 'seed_0/fit_targets.npz').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            p.verify_parent(self.work, manifest, HERE)

    def test_missing_file_not_dropped(self):
        (self.root / 'seed_0.log').unlink()
        with self.assertRaises(ValueError):
            self.manifest()

    def test_extra_file_rejected(self):
        (self.root / 'unregistered').write_text('extra')
        with self.assertRaises(ValueError):
            self.manifest()

    def test_symlink_rejected(self):
        path = self.root / 'seed_0/teacher_final.npz'
        path.unlink()
        path.symlink_to(self.root / 'seed_1/teacher_final.npz')
        with self.assertRaises(ValueError):
            self.manifest()

    def test_custody_tamper_not_sanitized_away(self):
        manifest = self.manifest()
        write(self.custody / 'seed_0.started.json', {'id': 0, 'changed': True})
        with self.assertRaises(ValueError):
            p.verify_parent(self.work, manifest, HERE)

    def test_nested_manifest_cross_binding(self):
        write(self.root / 'seed_0/FILES_SHA256.json', {})
        write(self.root / 'FILES_SHA256.json', {n: p.sha(self.root / n) for n in p.ARTIFACT_PATHS if n != 'FILES_SHA256.json'})
        with self.assertRaisesRegex(ValueError, 'nested manifest'):
            self.manifest()

    def test_historical_parent_relabelling_rejected(self):
        manifest = self.manifest()
        manifest['parent_contract'] = 'historical'
        with self.assertRaisesRegex(ValueError, 'historical'):
            p.verify_parent(self.work, manifest, HERE)

    def test_parent_summary_row_disagreement(self):
        row = p.read_json(self.root / 'seed_0/result.json')
        row['teacher_correct'] = 264
        write(self.root / 'seed_0/result.json', row)
        refresh(self.root, self.custody)
        with self.assertRaisesRegex(ValueError, 'summary/result'):
            self.manifest()

    def test_2048_pass_mismatch(self):
        row = p.read_json(self.root / 'seed_0/result.json')
        row['arms']['J']['quality_pass'] = False
        with self.assertRaisesRegex(ValueError, '2048 pass'):
            p.validate_reused_parent_row(row, row['seed'])

    def test_shared_schedule_and_600_steps_required(self):
        row = p.read_json(self.root / 'seed_0/result.json')
        row['fit_records']['J']['parameter_steps'][0] = 599
        with self.assertRaisesRegex(ValueError, 'final-step/shared-schedule'):
            p.validate_reused_parent_row(row, row['seed'])

    def test_parent_failed_exit_rejected_even_with_complete_rows(self):
        summary = p.read_json(self.root / 'SUMMARY.json')
        summary['attempts'][0]['returncode'] = 1
        write(self.root / 'SUMMARY.json', summary)
        refresh(self.root, self.custody)
        with self.assertRaisesRegex(ValueError, 'successful-exit'):
            self.manifest()

    def test_missing_private_approval_rejected(self):
        cohort = p.read_json(self.custody / 'cohort.json')
        cohort['release']['parent_approved'] = False
        write(self.custody / 'cohort.json', cohort)
        with self.assertRaisesRegex(ValueError, 'private approval'):
            self.manifest()

    def test_parent_finished_after_deadline_rejected(self):
        summary = p.read_json(self.root / 'SUMMARY.json')
        summary['finished_utc'] = '2026-01-02T00:00:00Z'
        write(self.root / 'SUMMARY.json', summary)
        refresh(self.root, self.custody)
        with self.assertRaisesRegex(ValueError, 'deadline'):
            self.manifest()

    def test_manifest_path_traversal_rejected(self):
        manifest = self.manifest()
        manifest['files']['../escape'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'path census'):
            p.verify_parent(self.work, manifest, HERE)

    def test_grid_release_exact_closure(self):
        value = release()
        self.assertEqual(g.validate_release(value, HERE, now_timestamp=0), value)
        value['files']['runner.py'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'file mismatch'):
            g.validate_release(value, HERE, now_timestamp=0)

    def test_grid_release_requires_parent_digest(self):
        value = release()
        value['parent_manifest_sha256'] = 'not a hash'
        with self.assertRaises(ValueError):
            g.validate_release(value, HERE, now_timestamp=0)

    def test_grid_context_namespace_and_no_existing_attempt_reuse(self):
        registry, out = g.context(self.work, 'replication-grid-synthetic', HERE)
        self.assertEqual(registry, self.work / 'custody/grid/replication-grid-synthetic')
        self.assertEqual(out, self.work / 'artifacts/replication-grid-synthetic')
        self.assertFalse(registry.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
