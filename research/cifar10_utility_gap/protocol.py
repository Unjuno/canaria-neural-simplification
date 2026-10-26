"""Standard-library-only, fail-closed protocol and launch/release gates.

Hashes establish bindings, not authority: a separately reviewed parent must supply
expected approval hashes out of band. This module never reads numerical artifacts.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time

BASE_COMMIT = 'bb30713cca06519a000251a26668309b1a4ef82b'
SEEDS = (26102601, 26102602, 26102603)
CELLS = ('w8_mse', 'w8_kl', 'w16_mse', 'w16_kl')
BUDGETS = (1024, 2048)
GLOBAL_DEADLINE = 1793256600.0  # 2026-10-29T06:50:00Z
FIT_CAP = 7200
COHORT_CAP = 21600
DIAGNOSTIC_CAP = 1800
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
DESIGN = dict(schema=1, experiment='canaria-utility-gap-v1', exploratory=True,
    formal_noninferiority=False, seeds=list(SEEDS), cells=list(CELLS), budgets=list(BUDGETS),
    fit_n=4000, diagnostic_n=1000, updates=2048, batch_size=128, passes=64,
    last_batch=32, span=[4, 6], widths=[8, 16], objectives=['mse', 'kl'], temperature=1,
    primary_contrast='w8_mse:1024->2048', optimizer=dict(name='Adam', lr=.001,
    betas=[.9,.999], eps=1e-8, weight_decay=0., clipping=False, schedule='constant'),
    fresh_initialization=True, warm_start=False, adaptation=False, official_test=False,
    original_validation=False, threads=4, interop=1, global_science_threads_max=8,
    fit_cap_seconds=FIT_CAP, diagnostic_cap_seconds=DIAGNOSTIC_CAP,
    cohort_cap_seconds=COHORT_CAP, global_deadline=GLOBAL_DEADLINE)


class DeadlineExceeded(TimeoutError):
    """Only an owned immutable clock/hard-deadline alarm has expired.

    Generic reader/OS TimeoutError is an operational failure, not this signal.
    """


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return sha(canonical(value))


def require_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected lowercase SHA256')
    return value


def ids_hash(ids):
    return sha(b''.join(struct.pack('<q', i) for i in ids))


def partition(calibration_ids, expected_calibration_sha256):
    ids = tuple(calibration_ids)
    if len(ids) != 5000 or len(set(ids)) != 5000 or any(type(i) is not int or not 0 <= i < 50000 for i in ids):
        raise ValueError('Exactly 5000 unique original training-namespace calibration IDs required')
    if ids_hash(ids) != require_hash(expected_calibration_sha256):
        raise ValueError('Original calibration ID hash mismatch')
    ordered = sorted(ids, key=lambda i: (sha(f'canaria-utility-gap-v1:{i}'.encode('utf-8')), i))
    diagnostic = tuple(ordered[:1000]); fit = tuple(ordered[1000:])
    return dict(schema=1, method='SHA256(canaria-utility-gap-v1:<decimal ID>), ID tie-break',
        calibration_ids=list(ids), calibration_sha256=ids_hash(ids),
        fit_ids=list(fit), diagnostic_ids=list(diagnostic),
        fit_sha256=ids_hash(fit), diagnostic_sha256=ids_hash(diagnostic),
        label_stratification=False, pristine_study_holdout=False)


def verify_partition(record):
    if record != partition(record['calibration_ids'], record['calibration_sha256']):
        raise ValueError('Partition differs from immutable deterministic rule')
    return digest(record)


def verify_original_lock(lock):
    if lock.get('status') != 'frozen' or lock.get('base_commit') != BASE_COMMIT:
        raise ValueError('Original cohort must finish and lock unchanged first')
    require_hash(lock.get('evaluator_identity_sha256'))
    records = lock.get('records', {})
    if set(records) != {str(s) for s in SEEDS}:
        raise ValueError('Every original planned seed, including failures, required')
    for record in records.values():
        status = record.get('status')
        if status not in ('complete', 'teacher-quality-failed', 'teacher-incomplete', 'fit-failed'):
            raise ValueError('Original cohort contains pending/unknown status')
        require_hash(record.get('terminal_record_sha256'))
        if status == 'complete':
            for key in ('teacher_sha256', 'replacement_sha256', 'validation_sha256', 'recipe_sha256'):
                require_hash(record.get(key))
        if record.get('original_negative_results_preserved') is not True:
            raise ValueError('Original evidence must remain untouched')
    return digest(lock)


def execution_gate(approval, expected_approval_sha256, original_lock, partition_record,
                   source_hashes, *, execution_authorized=False, now=None):
    if execution_authorized is not True:
        raise PermissionError('Scientific execution NOT GO: separate announced approval required')
    if digest(approval) != require_hash(expected_approval_sha256):
        raise PermissionError('Out-of-band announced approval binding mismatch')
    if approval.get('action') != 'approved-utility-gap-execution' or approval.get('design') != DESIGN:
        raise PermissionError('Exact separately reviewed fixed design required')
    if approval.get('original_cohort_sha256') != verify_original_lock(original_lock):
        raise ValueError('Original frozen-cohort binding mismatch')
    if approval.get('partition_sha256') != verify_partition(partition_record):
        raise ValueError('Publish and approve exact new partition before fitting')
    identity=approval.get('evaluator_identity', {})
    if digest(identity) != original_lock['evaluator_identity_sha256']:
        raise ValueError('Original approved evaluator identity mismatch')
    if identity.get('splits',{}).get('calibration') != dict(n=5000,sha256=partition_record['calibration_sha256']):
        raise ValueError('Partition not bound to original admitted calibration IDs')
    if not source_hashes or approval.get('source_sha256') != source_hashes:
        raise ValueError('Reviewed source binding mismatch')
    for value in source_hashes.values(): require_hash(value)
    if approval.get('parent_announced') is not True or approval.get('independent_review') != 'GO':
        raise PermissionError('Announced parent approval and independent source review required')
    if approval.get('fees') is not False or approval.get('network_writes') is not False or approval.get('ci') is not False:
        raise PermissionError('No fees, network writes or CI authorized')
    slot = approval.get('resource_slot', {})
    now = time.time() if now is None else now
    require_finite_time(now,'slot check time')
    require_finite_time(slot.get('starts'),'slot start')
    require_finite_time(slot.get('expires'),'slot expiry')
    if (not slot.get('reservation_id') or slot.get('threads') != 4 or slot.get('interop') != 1
            or type(slot.get('other_science_threads')) is not int
            or not 0 <= slot['other_science_threads'] <= 4
            or slot.get('original_cohort_has_priority') is not True
            or slot.get('single_worker_lock_required') is not True
            or not isinstance(slot.get('starts'), (int,float)) or not isinstance(slot.get('expires'), (int,float))
            or not slot['starts'] <= now < slot['expires'] <= GLOBAL_DEADLINE
            or slot['expires'] < now + FIT_CAP):
        raise PermissionError('Current 4-thread slot covering full two-hour allocation within global 8-thread cap required')
    return digest(approval)


def require_finite_time(value, label='time'):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('Finite non-boolean '+label+' required')
    return value


def write_exclusive_json(path, value):
    path = Path(path)
    with path.open('xb') as handle:
        handle.write(canonical(value)); handle.flush(); os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def open_cohort(directory, approval_sha256, *, now=None):
    """Called after execution_gate, BEFORE importing numerics or reading data."""
    now = time.time() if now is None else now
    require_finite_time(now,'cohort start')
    if not math.isfinite(now) or now > GLOBAL_DEADLINE - DIAGNOSTIC_CAP - FIT_CAP:
        raise TimeoutError('No complete two-hour fitting allocation plus diagnostic reserve remains')
    expiry = min(now + COHORT_CAP, GLOBAL_DEADLINE)
    record = dict(schema=1, approval_sha256=require_hash(approval_sha256), started=now,
        expiry=expiry, fit_cutoff=expiry-DIAGNOSTIC_CAP)
    write_exclusive_json(Path(directory)/'cohort_launch.json', record)
    return record


def verify_cohort_clock(record):
    require_hash(record['approval_sha256'])
    for key in ('started','expiry','fit_cutoff'): require_finite_time(record.get(key),'cohort '+key)
    if not math.isfinite(record['started']) or record['expiry'] != min(record['started']+COHORT_CAP, GLOBAL_DEADLINE):
        raise ValueError('Original cohort expiry changed')
    if record['fit_cutoff'] != record['expiry']-DIAGNOSTIC_CAP:
        raise ValueError('Diagnostic reserve removed')


def teacher_launch(directory, cohort, seed, *, now=None):
    verify_cohort_clock(cohort)
    now = time.time() if now is None else now
    require_finite_time(now,'teacher start')
    if seed not in SEEDS or not math.isfinite(now) or not cohort['started'] <= now < cohort['fit_cutoff']:
        raise TimeoutError('No remaining admitted fitting allocation')
    if now + FIT_CAP > cohort['fit_cutoff']:
        raise TimeoutError('Insufficient overall time for a full reserved teacher allocation; mark missing')
    record = dict(seed=seed, started=now, expiry=min(now+FIT_CAP, cohort['fit_cutoff']),
                  cohort_launch_sha256=digest(cohort))
    write_exclusive_json(Path(directory)/f'teacher_{seed}_launch.json', record)
    return record


def check_teacher_clock(record, cohort, *, now=None):
    verify_cohort_clock(cohort)
    now = time.time() if now is None else now
    require_finite_time(now,'teacher check time')
    for key in ('started','expiry'): require_finite_time(record.get(key),'teacher '+key)
    if record.get('seed') not in SEEDS or record.get('cohort_launch_sha256') != digest(cohort):
        raise ValueError('Teacher launch identity mismatch')
    if not cohort['started'] <= record['started'] < cohort['fit_cutoff']:
        raise ValueError('Teacher start outside original fitting window')
    if record['expiry'] != min(record['started']+FIT_CAP, cohort['fit_cutoff']):
        raise ValueError('Original teacher expiry changed')
    if now < record['started']:
        raise ValueError('Teacher check precedes immutable launch time')
    if now >= record['expiry']:
        raise DeadlineExceeded('Original teacher fitting cap expired')


def verify_trajectory(record):
    if record.get('status') not in ('complete', 'failed', 'cap-exhausted', 'not-admitted'):
        raise ValueError('All trajectories must be terminal before diagnostic release')
    cursor = record.get('updates')
    if type(cursor) is not int or not 0 <= cursor <= 2048:
        raise ValueError('Invalid terminal update count')
    if record['status'] == 'complete' and cursor != 2048:
        raise ValueError('Complete trajectory requires exactly 2048 updates')
    if record['status'] == 'not-admitted' and cursor != 0:
        raise ValueError('Not-admitted trajectory cannot have fitted updates')
    checkpoints = record.get('checkpoints', {})
    if set(checkpoints) != {'1024', '2048'}:
        raise ValueError('Both planned checkpoint statuses must be frozen')
    for step in BUDGETS:
        checkpoint = checkpoints[str(step)]
        if checkpoint.get('status') == 'saved':
            if cursor < step or record['status'] == 'not-admitted':
                raise ValueError('Checkpoint beyond actual trajectory')
            for key in ('checkpoint_sha256', 'fit_metrics_sha256'):
                require_hash(checkpoint.get(key))
        elif checkpoint.get('status') not in ('missing', 'failed'):
            raise ValueError('Unfrozen checkpoint status')
        elif not checkpoint.get('reason'):
            raise ValueError('Missing/failed checkpoints require explicit reason')
    if record['status'] == 'complete' and any(c['status'] != 'saved' for c in checkpoints.values()):
        raise ValueError('Complete trajectory missing prespecified metric/checkpoint')
    if record['status'] != 'complete' and not record.get('reason'):
        raise ValueError('Failure/not-admitted reason required')


def verify_frozen_cohort(frozen, partition_record, original_lock, source_hashes):
    if frozen.get('status') != 'all-trajectories-and-checkpoints-frozen':
        raise ValueError('Private diagnostic data remains sealed')
    if frozen.get('design_sha256') != digest(DESIGN) or frozen.get('partition_sha256') != verify_partition(partition_record):
        raise ValueError('Frozen design/partition mismatch')
    if frozen.get('original_cohort_sha256') != verify_original_lock(original_lock) or frozen.get('source_sha256') != source_hashes:
        raise ValueError('Frozen original/source binding mismatch')
    records = frozen.get('teachers', {})
    if set(records) != {str(s) for s in SEEDS}: raise ValueError('All three planned teachers required')
    for seed, record in records.items():
        if set(record.get('trajectories', {})) != set(CELLS):
            raise ValueError('All four prespecified trajectories required for each teacher')
        if record.get('admission') not in ('admitted', 'not-admitted'):
            raise ValueError('Teacher admission status missing')
        if record['admission'] == 'admitted':
            original = original_lock['records'][seed]
            if original['status'] in ('teacher-quality-failed', 'teacher-incomplete'):
                raise ValueError('Original teacher failed admission; no substitute allowed')
            if record.get('teacher_sha256') != original.get('teacher_sha256'):
                raise ValueError('Follow-on teacher differs from original locked teacher')
            require_hash(record.get('teacher_sha256'))
            before = require_hash(record.get('teacher_state_before_sha256'))
            if record.get('teacher_state_after_sha256') != before:
                raise ValueError('Teacher state mutated; diagnostics denied')
        for trajectory in record['trajectories'].values():
            verify_trajectory(trajectory)
            if (record['admission'] == 'not-admitted') != (trajectory['status'] == 'not-admitted'):
                raise ValueError('Teacher and trajectory admission disagree')
    return digest(frozen)


def release_gate(approval, expected_approval_sha256, frozen, partition_record, original_lock,
                 source_hashes, *, release_authorized=False):
    if release_authorized is not True:
        raise PermissionError('Separate private diagnostic release approval required')
    frozen_sha = verify_frozen_cohort(frozen, partition_record, original_lock, source_hashes)
    if digest(approval) != require_hash(expected_approval_sha256):
        raise PermissionError('Diagnostic release approval binding mismatch')
    if approval != dict(action='approved-private-utility-gap-diagnostic-release',
                        frozen_cohort_sha256=frozen_sha, official_test=False,
                        original_validation=False, exploratory=True):
        raise PermissionError('Exact frozen full-cohort private release required')
    return frozen_sha


def begin_release(directory, cohort, frozen_sha256, *, now=None):
    """Exclusive consumed marker BEFORE any diagnostic images/labels are read.

    A crash leaves an incomplete one-look result, never permission for a retry.
    """
    verify_cohort_clock(cohort)
    now = time.time() if now is None else now
    require_finite_time(now,'release start')
    if not cohort['started'] <= now < cohort['expiry']:
        raise TimeoutError('Overall six-hour/global deadline expired')
    record = dict(schema=1, frozen_cohort_sha256=require_hash(frozen_sha256), started=now,
        expiry=min(now+DIAGNOSTIC_CAP, cohort['expiry']), cohort_launch_sha256=digest(cohort))
    write_exclusive_json(Path(directory)/'diagnostic_consumed.json', record)
    return record


def check_release_clock(record, cohort, *, now=None):
    verify_cohort_clock(cohort)
    now = time.time() if now is None else now
    require_finite_time(now,'release check time')
    for key in ('started','expiry'): require_finite_time(record.get(key),'release '+key)
    if record['cohort_launch_sha256'] != digest(cohort) or record['expiry'] != min(record['started']+DIAGNOSTIC_CAP, cohort['expiry']):
        raise ValueError('Diagnostic release cap changed')
    if not cohort['started'] <= record['started'] <= now:
        raise ValueError('Diagnostic check precedes immutable release time')
    if now >= record['expiry']:
        raise DeadlineExceeded('One-look diagnostic cap expired')
