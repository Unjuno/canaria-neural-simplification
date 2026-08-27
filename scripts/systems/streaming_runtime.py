from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil

D = 1024
R = 64
N_BLOCKS = 32
BATCH = 8
SEED = 20260827
ALPHA = np.float32(0.05)
REPEATS = 3
RSS_INTERVAL_S = 0.001
MODEL_BUDGET_BYTES = 32 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def generate_artifacts(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((BATCH, D), dtype=np.float32)
    np.save(root / 'input.npy', x)

    dense_bytes = 0
    factor_bytes = 0
    for i in range(N_BLOCKS):
        u = (rng.standard_normal((D, R), dtype=np.float32) / np.float32(math.sqrt(D))).astype(np.float32, copy=False)
        v = (rng.standard_normal((R, D), dtype=np.float32) / np.float32(math.sqrt(R))).astype(np.float32, copy=False)
        w = (u @ v).astype(np.float32, copy=False)
        up = root / f'block_{i:03d}_U.npy'
        vp = root / f'block_{i:03d}_V.npy'
        wp = root / f'block_{i:03d}_W.npy'
        np.save(up, u)
        np.save(vp, v)
        np.save(wp, w)
        factor_bytes += up.stat().st_size + vp.stat().st_size
        dense_bytes += wp.stat().st_size
        del u, v, w

    manifest = {
        'format': 'canaria-systems-s1-v1',
        'd': D,
        'r': R,
        'blocks': N_BLOCKS,
        'batch': BATCH,
        'seed': SEED,
        'alpha': float(ALPHA),
        'dtype': 'float32',
        'dense_serialized_bytes': dense_bytes,
        'factor_serialized_bytes': factor_bytes,
        'input_sha256': sha256_file(root / 'input.npy'),
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


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


def execute_dense_resident(root: Path, x0: np.ndarray) -> tuple[np.ndarray, int, list[float]]:
    weights = [np.load(root / f'block_{i:03d}_W.npy') for i in range(N_BLOCKS)]
    payload = sum(int(w.nbytes) for w in weights)
    times = []
    out = None
    for _ in range(REPEATS):
        x = x0.copy()
        t0 = time.perf_counter()
        for w in weights:
            x = x + ALPHA * (x @ w)
        times.append(time.perf_counter() - t0)
        out = x
    assert out is not None
    return out, payload, times


def execute_dense_streaming(root: Path, x0: np.ndarray) -> tuple[np.ndarray, int, list[float]]:
    one_payload = D * D * np.dtype(np.float32).itemsize
    times = []
    out = None
    for _ in range(REPEATS):
        x = x0.copy()
        t0 = time.perf_counter()
        for i in range(N_BLOCKS):
            w = np.load(root / f'block_{i:03d}_W.npy', mmap_mode='r')
            x = x + ALPHA * (x @ w)
            del w
        times.append(time.perf_counter() - t0)
        out = x
    assert out is not None
    return out, int(one_payload), times


def execute_factor_streaming(root: Path, x0: np.ndarray) -> tuple[np.ndarray, int, list[float]]:
    one_payload = 2 * D * R * np.dtype(np.float32).itemsize
    times = []
    out = None
    for _ in range(REPEATS):
        x = x0.copy()
        t0 = time.perf_counter()
        for i in range(N_BLOCKS):
            u = np.load(root / f'block_{i:03d}_U.npy', mmap_mode='r')
            v = np.load(root / f'block_{i:03d}_V.npy', mmap_mode='r')
            tmp = x @ u
            x = x + ALPHA * (tmp @ v)
            del tmp, u, v
        times.append(time.perf_counter() - t0)
        out = x
    assert out is not None
    return out, int(one_payload), times


def probe(root: Path, mode: str, output_path: Path) -> dict:
    x0 = np.load(root / 'input.npy')
    _ = float(x0.sum())
    gc.collect()
    proc = psutil.Process(os.getpid())
    baseline = proc.memory_info().rss
    stop, th, samples = start_rss_sampler(proc)
    try:
        if mode == 'dense_resident':
            out, payload, times = execute_dense_resident(root, x0)
        elif mode == 'dense_streaming':
            out, payload, times = execute_dense_streaming(root, x0)
        elif mode == 'factor_streaming':
            out, payload, times = execute_factor_streaming(root, x0)
        else:
            raise ValueError(mode)
        np.save(output_path, out)
    finally:
        stop.set()
        th.join(timeout=2.0)

    peak = max(samples) if samples else proc.memory_info().rss
    return {
        'mode': mode,
        'baseline_rss_bytes': int(baseline),
        'peak_rss_bytes': int(peak),
        'peak_rss_delta_bytes': int(max(0, peak - baseline)),
        'peak_model_payload_bytes_by_construction': int(payload),
        'inference_seconds': [float(x) for x in times],
        'median_inference_seconds': float(statistics.median(times)),
        'output_sha256': sha256_file(output_path),
    }


def run_child(script: Path, root: Path, mode: str, out_dir: Path) -> dict:
    output_path = out_dir / f'{mode}_output.npy'
    env = dict(os.environ)
    env.update({
        'OPENBLAS_NUM_THREADS': '1',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
    })
    text = subprocess.check_output(
        [sys.executable, str(script), '--probe-mode', mode, '--artifact-dir', str(root), '--probe-output', str(output_path)],
        text=True,
        env=env,
    )
    return json.loads(text)


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a.astype(np.float64))) + 1e-30
    return float(np.linalg.norm((a - b).astype(np.float64)) / denom)


def orchestrate(artifact_dir: Path, out_dir: Path, report_path: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = generate_artifacts(artifact_dir)
    script = Path(__file__).resolve()
    modes = ['dense_resident', 'dense_streaming', 'factor_streaming']
    probes = {m: run_child(script, artifact_dir, m, out_dir) for m in modes}

    dense = np.load(out_dir / 'dense_resident_output.npy')
    dense_stream = np.load(out_dir / 'dense_streaming_output.npy')
    factor = np.load(out_dir / 'factor_streaming_output.npy')

    dense_stream_max_abs = float(np.max(np.abs(dense_stream - dense)))
    factor_max_abs = float(np.max(np.abs(factor - dense)))
    factor_rel_l2 = relative_l2(dense, factor)

    resident_payload = probes['dense_resident']['peak_model_payload_bytes_by_construction']
    factor_payload = probes['factor_streaming']['peak_model_payload_bytes_by_construction']
    capacity_witness = {
        'budget_bytes': MODEL_BUDGET_BYTES,
        'dense_resident_exceeds_budget': bool(resident_payload > MODEL_BUDGET_BYTES),
        'factor_streaming_fits_budget': bool(factor_payload < MODEL_BUDGET_BYTES),
    }

    checks = {
        'factor_payload_ratio_lte_0_20': bool(factor_payload / resident_payload <= 0.20),
        'factor_rss_delta_lt_dense_resident': bool(probes['factor_streaming']['peak_rss_delta_bytes'] < probes['dense_resident']['peak_rss_delta_bytes']),
        'factor_max_abs_lte_0_001': bool(factor_max_abs <= 0.001),
        'factor_relative_l2_lte_0_0001': bool(factor_rel_l2 <= 0.0001),
        'dense_stream_max_abs_lte_1e_6': bool(dense_stream_max_abs <= 1e-6),
        'capacity_witness': bool(capacity_witness['dense_resident_exceeds_budget'] and capacity_witness['factor_streaming_fits_budget']),
    }

    report = {
        'experiment': 'Canaria Systems S1 streaming compact runtime',
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'evidence_class': 'systems_runtime_format_poc',
        'scientific_claim_use': 'DO_NOT_USE_AS_COMPOSITION_GENERALIZATION_EVIDENCE',
        'environment': {
            'python': platform.python_version(),
            'platform': platform.platform(),
            'numpy': np.__version__,
            'psutil': psutil.__version__,
            'blas_thread_env_for_probes': 1,
        },
        'manifest': manifest,
        'probes': probes,
        'output_agreement': {
            'dense_streaming_vs_dense_resident_max_abs': dense_stream_max_abs,
            'factor_streaming_vs_dense_resident_max_abs': factor_max_abs,
            'factor_streaming_vs_dense_resident_relative_l2': factor_rel_l2,
        },
        'derived': {
            'factor_peak_model_payload_ratio_vs_dense_resident': factor_payload / resident_payload,
            'factor_peak_rss_delta_ratio_vs_dense_resident': (
                probes['factor_streaming']['peak_rss_delta_bytes'] / probes['dense_resident']['peak_rss_delta_bytes']
                if probes['dense_resident']['peak_rss_delta_bytes'] else None
            ),
            'dense_streaming_peak_rss_delta_ratio_vs_dense_resident': (
                probes['dense_streaming']['peak_rss_delta_bytes'] / probes['dense_resident']['peak_rss_delta_bytes']
                if probes['dense_resident']['peak_rss_delta_bytes'] else None
            ),
            'factor_serialized_ratio_vs_dense': manifest['factor_serialized_bytes'] / manifest['dense_serialized_bytes'],
        },
        'capacity_witness': capacity_witness,
        'checks': checks,
        'interpretation_boundary': 'Synthetic exact low-rank operator stack. Demonstrates a runtime representation/materialization mechanism only; it does not establish that arbitrary trained neural spans admit this representation or that any specific MCU/edge device can run it.',
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=Path('systems_s1_out'))
    ap.add_argument('--report', type=Path, default=Path('systems_s1_report.json'))
    ap.add_argument('--probe-mode', choices=['dense_resident', 'dense_streaming', 'factor_streaming'])
    ap.add_argument('--probe-output', type=Path)
    args = ap.parse_args()

    if args.probe_mode:
        if args.probe_output is None:
            raise SystemExit('--probe-output is required with --probe-mode')
        print(json.dumps(probe(args.artifact_dir, args.probe_mode, args.probe_output)))
    else:
        orchestrate(args.artifact_dir, args.out_dir, args.report)


if __name__ == '__main__':
    main()
