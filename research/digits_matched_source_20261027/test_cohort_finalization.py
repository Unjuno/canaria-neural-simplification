"""Pure fake-clock coordinator regression; no numerical/runner/process imports.

Run this file directly beside the co-published matched and grid packages. Only
the reviewed cohort function is AST-compiled. Workers, clock, analysis and
resource probes are fake; opaque fixture bytes are not models or data. Fake
release metadata is not scientific execution authority. Temporary fixture
custody never touches an actual attempt. This is not end-to-end replication.
"""
import ast
import datetime
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import types

MATCHED = pathlib.Path(__file__).resolve().parent
GRID = MATCHED.parent / 'digits_joint_budget_source_20261027'


def load_contract(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load_contract('fake_finalization_matched_contract', MATCHED / 'portable_contract.py')
p = load_contract('fake_finalization_parent_contract', GRID / 'parent_contract.py')
tree = ast.parse((MATCHED / 'runner.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'cohort')
cohort_code = compile(ast.Module(body=[node], type_ignores=[]), str(MATCHED / 'runner.py'), 'exec')
seed_files = set(json.loads((MATCHED / 'ARTIFACT_SCHEMA.json').read_text())['x-per-seed-files']) - {'FAILURE.json', 'FILES_SHA256.json'}
seeds = json.loads((MATCHED / 'SEEDS.json').read_text())['seeds']


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)


class Clock:
    def __init__(self):
        self.value = p.timestamp('2026-01-01T00:00:00Z')
        self.analysis_finished = False
        self.finalization_captures = 0

    def time(self):
        assert not self.analysis_finished, 'finalization must use its one captured UTC value'
        return self.value

    def sleep(self, unused):
        self.value += .001

    def now(self):
        if self.analysis_finished:
            self.finalization_captures += 1
            assert self.finalization_captures == 1, 'completion clock sampled more than once'
        return datetime.datetime.fromtimestamp(self.value, datetime.timezone.utc).isoformat()


def parent_outcome(work, identity):
    try:
        descriptor = p.validate_completed_parent(work, identity, GRID)
        assert len(descriptor['files']) == 339 and len(descriptor['custody_files']) == 19
        return {'status': 'ACCEPT', 'artifact_files': 339, 'custody_files': 19}
    except ValueError as exc:
        return {'status': 'REJECT', 'reason': str(exc)}


def scenario(label, offset, decision='NOT_JOINTLY_SUPPORTED', first_worker_fails=False):
    with tempfile.TemporaryDirectory(prefix='canaria-artificial-finalization-') as tmp:
        work = pathlib.Path(tmp)
        identity = 'replication-artificial-finalization'
        registry, out = m.context(work, identity, MATCHED)
        clock = Clock()
        deadline = clock.value + 10
        release = {
            'contract': m.CONTRACT, 'replication_id': identity,
            'evidence_class': m.EVIDENCE_CLASS, 'historical_attempt': m.HISTORICAL_ATTEMPT,
            'custody_attestation': m.CUSTODY_ATTESTATION, 'parent_approved': True,
            'independent_review': 'GO', 'reviewer': 'ARTIFICIAL FIXTURE NOT AUTHORIZATION',
            'approval_evidence': 'ARTIFICIAL FIXTURE NOT AUTHORIZATION',
            'package_sha256': m.package_identity(MATCHED), 'files': m.manifest_hashes(MATCHED),
            'deadline_utc': datetime.datetime.fromtimestamp(deadline, datetime.timezone.utc).isoformat(),
        }
        release_path = work / 'ARTIFICIAL_RELEASE.json'
        dump(release_path, release)
        worker_ids, rows_seen = [], []

        class FakeProcess:
            def __init__(self, command, **unused):
                i = int(command[command.index('--id') + 1])
                worker_ids.append(i)
                self.pid, self.returncode = i + 1, int(first_worker_fails and i == 0)
                seed_root = out / f'seed_{i}'
                seed_root.mkdir()
                for name in seed_files - {'result.json'}:
                    (seed_root / name).write_bytes(b'ARTIFICIAL OPAQUE NONMODEL NONDATA BYTES')
                row = {
                    'id': i, 'seed': seeds[i], 'status': 'COMPLETE', 'teacher_correct': 260,
                    'teacher_initial_sha256': 'a' * 64, 'teacher_final_sha256': 'b' * 64, 'schedule_sha256': 'c' * 64,
                    'arms': {a: {'nmse': .06, 'correct': 258, 'n': 270, 'accuracy': 258 / 270, 'quality_pass': True, 'state_sha256': 'd' * 64} for a in ('L', 'J', 'C')},
                    'fit_records': {a: {'parameter_steps': [600] * (2 if a == 'C' else 4), 'examples_per_parameter': 76800, 'schedule_sha256': 'c' * 64} for a in ('L', 'J', 'C')},
                }
                dump(seed_root / 'result.json', row)
                dump(seed_root / 'FILES_SHA256.json', {x.name: m.sha(x) for x in sorted(seed_root.iterdir()) if x.is_file()})

            def poll(self):
                return self.returncode

        def claim(path, record):
            path.mkdir(parents=True, exist_ok=False)
            dump(path / 'claim.json', record)

        def analyze(rows):
            rows_seen.extend(rows)
            clock.value = deadline + offset
            clock.analysis_finished = True
            return {'decision': decision, 'all_rows': rows}

        namespace = {
            'contract': m, 'validate_release': lambda value: m.validate_release(value, MATCHED, now_timestamp=clock.time()),
            'REGISTRY': registry, 'DEADLINE': deadline, 'WORKSPACE': work, 'REPLICATION_ID': identity,
            'HERE': MATCHED, 'pathlib': pathlib, 'json': json, 'hashlib': hashlib,
            'claim_attempt': claim, 'dump': dump, 'now': clock.now,
            'source_identity': lambda: m.package_identity(MATCHED), 'sha': m.sha,
            'os': types.SimpleNamespace(environ={}, urandom=lambda n: b'x' * n, getpid=lambda: 0),
            'time': clock, 'sys': types.SimpleNamespace(executable='FAKE NO EXECUTION'),
            '__file__': str(MATCHED / 'runner.py'),
            'subprocess': types.SimpleNamespace(Popen=FakeProcess, STDOUT=None, TimeoutExpired=TimeoutError),
            'tree_bytes': lambda unused: 0, 'rss_bytes': lambda unused: 0, 'analyze': analyze,
        }
        exec(cohort_code, namespace)
        summary = namespace['cohort'](release_path, out)
        finished = json.loads((registry / 'FINISHED.json').read_text())
        assert clock.finalization_captures == 1
        assert summary['finished_utc'] == finished['utc']
        assert p.timestamp(summary['finished_utc']) == deadline + offset
        assert finished['decision'] == summary['decision']
        assert finished['summary_sha256'] == m.sha(out / 'SUMMARY.json')
        assert json.loads((registry / 'cohort.json').read_text())['end_timestamp'] == deadline
        assert json.loads((out / 'run_contract.json').read_text())['deadline_utc'] == release['deadline_utc']
        if not first_worker_fails:
            assert worker_ids == list(range(16))
            assert len(summary['attempts']) == 16 and summary['not_started'] == []
            assert {r['id'] for r in summary['attempts']} == set(range(16))
            assert all(r['returncode'] == 0 and r['reason'] is None for r in summary['attempts'])
            assert len(rows_seen) == 16
        result = {
            'case': label, 'producer_decision': summary['decision'], 'abort_reason': summary['abort_reason'],
            'fake_workers': len(worker_ids), 'finalization_captures': clock.finalization_captures,
            'finished_utc': summary['finished_utc'], 'parent_adapter': parent_outcome(work, identity),
        }
        if first_worker_fails:
            assert summary['decision'] == 'INCOMPLETE' and summary['abort_reason'] == 'seed failed'
            assert summary['reason'] == 'seed failed' and summary['not_started'] == list(range(2, 16))
        elif offset <= 0:
            assert summary['decision'] == decision and summary['abort_reason'] is None
            assert summary['all_rows'] == rows_seen and result['parent_adapter']['status'] == 'ACCEPT'
        else:
            assert summary['decision'] == 'INCOMPLETE'
            assert summary['abort_reason'] == summary['reason'] == 'cohort deadline/cap during finalization'
            assert summary['completed_results'] == 16 and summary['no_exclusion_or_replacement'] is True
            assert result['parent_adapter'] == {'status': 'REJECT', 'reason': 'parent is not a complete matched cohort'}
            # Only mutate artificial metadata to prove the unchanged consumer
            # also rejects the old late-but-complete producer shape.
            forged = dict(summary, decision=decision, abort_reason=None, all_rows=rows_seen)
            (out / 'SUMMARY.json').write_text(json.dumps(forged, sort_keys=True))
            finished.update(decision=decision, summary_sha256=m.sha(out / 'SUMMARY.json'))
            (registry / 'FINISHED.json').write_text(json.dumps(finished, sort_keys=True))
            top = {str(x.relative_to(out)): m.sha(x) for x in sorted(out.rglob('*')) if x.is_file() and x != out / 'FILES_SHA256.json'}
            (out / 'FILES_SHA256.json').write_text(json.dumps(top, sort_keys=True))
            result['forged_complete_late_adapter'] = parent_outcome(work, identity)
            assert result['forged_complete_late_adapter'] == {'status': 'REJECT', 'reason': 'parent completion deadline binding mismatch'}
        return result


if __name__ == '__main__':
    results = [
        scenario('on_time_complete_negative', -1),
        scenario('exact_boundary_complete_negative', 0),
        scenario('late_analysis_all_16_successful_exits', 1),
        scenario('on_time_all_16_complete_positive', -1, 'CONDITIONAL_REGIME_SUPPORTED'),
        scenario('late_finalization_preserves_first_abort', 1, first_worker_fails=True),
    ]
    assert not {'numpy', 'torch', 'sklearn', 'scipy', 'subprocess'} & set(sys.modules)
    print(json.dumps({
        'status': 'PASS', 'cases': results,
        'scientific_execution': False, 'real_child_processes': False,
        'real_clocks_used_by_coordinator': False, 'numerical_imports': False,
        'runner_imports': False, 'actual_attempts_created': False,
    }, indent=2, sort_keys=True))
