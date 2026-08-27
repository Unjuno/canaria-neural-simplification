from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import sklearn
import torch

EXPECTED_ENV = {
    'python': '3.13.5',
    'torch': '2.10.0+cpu',
    'numpy': '2.3.5',
    'scikit_learn': '1.8.0',
    'psutil': '7.2.2',
}
BLOCK_HASHES = {
    'compiler_block0.pt': 'abb2ac470084721948a3521075f1a7f18d08ea61611a3f8a379f8b77d6719daa',
    'compiler_block1.pt': '7bd667e433e93d9b6e18484bf7d668c95dd5c317c139480a47c1edd20e3c35ca',
}
LOGICAL_BLOCK_CHUNKS = 4096
EXPECTED_CHECKSUM = 234473.0256500244
HEADROOM_BYTES = 64 * 1024 * 1024
RSS_INTERVAL_S = 0.001

torch.set_num_threads(1)


def current_env() -> dict:
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'numpy': np.__version__,
        'scikit_learn': sklearn.__version__,
        'psutil': psutil.__version__,
        'platform': platform.platform(),
        'torch_threads': torch.get_num_threads(),
    }


def env_matches() -> bool:
    cur = current_env()
    return all(cur[k] == v for k, v in EXPECTED_ENV.items()) and cur['torch_threads'] == 1 and sys.platform.startswith('linux')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_payload(artifact_dir: Path) -> dict:
    observed = {name: sha256_file(artifact_dir / name) for name in BLOCK_HASHES}
    ok = observed == BLOCK_HASHES
    return {'ok': ok, 'expected': BLOCK_HASHES, 'observed': observed}


def touch_state(state: dict) -> float:
    s = 0.0
    for v in state.values():
        if torch.is_tensor(v):
            s += float(v.sum())
    return s


def start_rss_sampler(proc: psutil.Process):
    stop = threading.Event()
    samples: list[int] = []

    def worker():
        while not stop.is_set():
            try:
                samples.append(proc.memory_info().rss)
            except psutil.Error:
                break
            time.sleep(RSS_INTERVAL_S)
        try:
            samples.append(proc.memory_info().rss)
        except psutil.Error:
            pass

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    return stop, th, samples


def memory_failure(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = f'{type(exc).__name__}: {exc}'.lower()
    keys = ['memory', 'alloc', 'bad_alloc', 'cannot allocate', 'out of memory', 'not enough memory']
    return any(k in text for k in keys)


def child_run(artifact_dir: Path, mode: str) -> dict:
    env = current_env()
    payload_check = verify_payload(artifact_dir)
    if not env_matches():
        return {'status': 'ENVIRONMENT_MISMATCH', 'environment': env, 'payload_check': payload_check}
    if not payload_check['ok']:
        return {'status': 'PAYLOAD_HASH_MISMATCH', 'environment': env, 'payload_check': payload_check}
    if not hasattr(resource, 'RLIMIT_AS'):
        return {'status': 'RLIMIT_AS_UNAVAILABLE', 'environment': env, 'payload_check': payload_check}

    # Warm up serialization/allocator paths before defining the constrained headroom.
    warm = torch.load(artifact_dir / 'compiler_block0.pt', map_location='cpu', weights_only=True)
    warm_checksum = touch_state(warm)
    del warm
    gc.collect()

    proc = psutil.Process(os.getpid())
    stop, th, samples = start_rss_sampler(proc)
    time.sleep(0.005)
    base_mem = proc.memory_info()
    baseline_rss = int(base_mem.rss)
    baseline_vms = int(base_mem.vms)
    constrained = mode in ('full_constrained', 'streaming_constrained')
    limit_bytes = None
    old_limit = resource.getrlimit(resource.RLIMIT_AS)
    if constrained:
        limit_bytes = baseline_vms + HEADROOM_BYTES
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))

    retained = []
    checksum = 0.0
    completed = 0
    error = None
    status = 'SUCCESS'
    t0 = time.perf_counter()
    try:
        for i in range(LOGICAL_BLOCK_CHUNKS):
            p = artifact_dir / f'compiler_block{i % 2}.pt'
            state = torch.load(p, map_location='cpu', weights_only=True)
            checksum += touch_state(state)
            if mode in ('full_unconstrained_control', 'full_constrained'):
                retained.append(state)
            elif mode == 'streaming_constrained':
                del state
                if (i + 1) % 128 == 0:
                    gc.collect()
            else:
                raise ValueError(mode)
            completed = i + 1
    except BaseException as exc:
        error = f'{type(exc).__name__}: {exc}'
        status = 'MEMORY_ALLOCATION_FAILURE' if memory_failure(exc) else 'UNRELATED_FAILURE'
    elapsed = time.perf_counter() - t0

    stop.set()
    th.join(timeout=2.0)
    try:
        final_mem = proc.memory_info()
        peak_rss = max(samples) if samples else final_mem.rss
        final_vms = int(final_mem.vms)
    except psutil.Error:
        peak_rss = max(samples) if samples else baseline_rss
        final_vms = None

    return {
        'status': status,
        'mode': mode,
        'environment': env,
        'payload_check': payload_check,
        'warm_checksum': warm_checksum,
        'logical_block_chunks_target': LOGICAL_BLOCK_CHUNKS,
        'completed_chunks': completed,
        'checksum': checksum,
        'expected_checksum': EXPECTED_CHECKSUM,
        'elapsed_seconds': elapsed,
        'baseline_rss_bytes': baseline_rss,
        'peak_rss_bytes': int(peak_rss),
        'peak_rss_delta_bytes': int(max(0, peak_rss - baseline_rss)),
        'baseline_vms_bytes': baseline_vms,
        'final_vms_bytes': final_vms,
        'constrained': constrained,
        'headroom_bytes': HEADROOM_BYTES if constrained else None,
        'rlimit_as_bytes': limit_bytes,
        'old_rlimit_as': [old_limit[0], old_limit[1]],
        'error': error,
    }


def run_child(script: Path, artifact_dir: Path, mode: str) -> dict:
    env = dict(os.environ)
    env.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
    })
    p = subprocess.run(
        [sys.executable, str(script), '--artifact-dir', str(artifact_dir), '--child-mode', mode],
        text=True,
        capture_output=True,
        env=env,
    )
    parsed = None
    if p.stdout.strip():
        try:
            parsed = json.loads(p.stdout.strip().splitlines()[-1])
        except json.JSONDecodeError:
            parsed = None
    return {
        'returncode': p.returncode,
        'stdout': p.stdout,
        'stderr': p.stderr,
        'result': parsed,
    }


def checksum_ok(x: float | None) -> bool:
    return x is not None and abs(float(x) - EXPECTED_CHECKSUM) <= 1e-6


def orchestrate(artifact_dir: Path, report_path: Path) -> dict:
    env = current_env()
    payload_check = verify_payload(artifact_dir)
    if not env_matches() or not payload_check['ok'] or not hasattr(resource, 'RLIMIT_AS'):
        status = 'ENVIRONMENT_OR_PAYLOAD_UNSUPPORTED'
        report = {'experiment': 'Canaria Systems S3 constrained-memory feasibility', 'status': status, 'environment': env, 'payload_check': payload_check, 'has_rlimit_as': hasattr(resource, 'RLIMIT_AS')}
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        return report

    script = Path(__file__).resolve()
    modes = ['full_unconstrained_control', 'full_constrained', 'streaming_constrained']
    runs = {m: run_child(script, artifact_dir, m) for m in modes}

    control = runs['full_unconstrained_control']['result']
    full_c = runs['full_constrained']['result']
    stream_c = runs['streaming_constrained']['result']

    control_ok = bool(
        runs['full_unconstrained_control']['returncode'] == 0 and control and control.get('status') == 'SUCCESS'
        and control.get('completed_chunks') == LOGICAL_BLOCK_CHUNKS and checksum_ok(control.get('checksum'))
    )
    stream_ok = bool(
        runs['streaming_constrained']['returncode'] == 0 and stream_c and stream_c.get('status') == 'SUCCESS'
        and stream_c.get('completed_chunks') == LOGICAL_BLOCK_CHUNKS and checksum_ok(stream_c.get('checksum'))
        and stream_c.get('headroom_bytes') == HEADROOM_BYTES
    )
    full_failed_for_memory = bool(
        runs['full_constrained']['returncode'] == 0 and full_c
        and full_c.get('status') == 'MEMORY_ALLOCATION_FAILURE'
        and int(full_c.get('completed_chunks', LOGICAL_BLOCK_CHUNKS)) < LOGICAL_BLOCK_CHUNKS
        and full_c.get('headroom_bytes') == HEADROOM_BYTES
    )

    checks = {
        'unconstrained_full_control_completes': control_ok,
        'constrained_streaming_completes': stream_ok,
        'constrained_full_fails_for_memory': full_failed_for_memory,
        'same_locked_headroom_used_for_constrained_modes': bool(full_c and stream_c and full_c.get('headroom_bytes') == HEADROOM_BYTES and stream_c.get('headroom_bytes') == HEADROOM_BYTES),
    }
    report = {
        'experiment': 'Canaria Systems S3 constrained-memory feasibility',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'evidence_class': 'systems_constrained_process_feasibility',
        'scientific_claim_use': 'DO_NOT_USE_AS_COMPOSITION_GENERALIZATION_EVIDENCE',
        'environment': env,
        'payload_check': payload_check,
        'locked_extra_address_space_headroom_bytes': HEADROOM_BYTES,
        'runs': runs,
        'checks': checks,
        'interpretation_boundary': 'Explicit Linux RLIMIT_AS feasibility boundary for the S2 learned-payload amplification harness. Not a physical-device deployment result; not a claim that the 4096 logical chunks form a trained model.',
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact-dir', type=Path, required=True)
    ap.add_argument('--report', type=Path, default=Path('systems_s3_report.json'))
    ap.add_argument('--child-mode', choices=['full_unconstrained_control', 'full_constrained', 'streaming_constrained'])
    args = ap.parse_args()
    if args.child_mode:
        print(json.dumps(child_run(args.artifact_dir, args.child_mode)))
    else:
        orchestrate(args.artifact_dir, args.report)


if __name__ == '__main__':
    main()
