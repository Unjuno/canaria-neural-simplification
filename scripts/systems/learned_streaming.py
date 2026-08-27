from __future__ import annotations

import argparse
import gc
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
import sklearn
import torch
import torch.nn as nn

EXPECTED_ENV = {
    'python': '3.13.5',
    'torch': '2.10.0+cpu',
    'numpy': '2.3.5',
    'scikit_learn': '1.8.0',
    'psutil': '7.2.2',
}
SEED = 4300
LOGICAL_BLOCK_CHUNKS = 4096
RSS_INTERVAL_S = 0.001


def add_g7_import_path() -> None:
    here = Path(__file__).resolve()
    repo = here.parents[2]
    g7 = repo / 'scripts' / 'reproduce' / 'g7_confirmatory'
    sys.path.insert(0, str(g7))


add_g7_import_path()
import run_seed as g  # noqa: E402
import runtime_poc as rp  # noqa: E402

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
    return all(cur[k] == v for k, v in EXPECTED_ENV.items()) and cur['torch_threads'] == 1


def tensor_payload_bytes(state: dict) -> int:
    return int(sum(v.numel() * v.element_size() for v in state.values() if torch.is_tensor(v)))


def package_learned_compact(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    _large, compact = rp.build_models(SEED)
    compact.eval()
    assert compact.compiler is not None
    assert len(compact.compiler.blocks) == 2

    full_state = {k: v.detach().cpu() for k, v in compact.state_dict().items()}
    shell_state = {k: v for k, v in full_state.items() if not k.startswith('compiler.blocks.')}
    block_states = [
        {k: v.detach().cpu() for k, v in compact.compiler.blocks[i].state_dict().items()}
        for i in range(2)
    ]

    full_path = out_dir / 'compact_full.pt'
    shell_path = out_dir / 'shell.pt'
    block_paths = [out_dir / 'compiler_block0.pt', out_dir / 'compiler_block1.pt']
    torch.save(full_state, full_path)
    torch.save(shell_state, shell_path)
    for state, path in zip(block_states, block_paths):
        torch.save(state, path)

    tr, va, te = g.datasets()
    historical = g.tf_metrics(compact, te)
    del tr, va

    manifest = {
        'format': 'canaria-systems-s2-learned-stream-v1',
        'source_seed': SEED,
        'source': 'G7 progressive 4->3->2 learned compact compiler',
        'full_state_file': full_path.name,
        'shell_file': shell_path.name,
        'block_files': [p.name for p in block_paths],
        'full_tensor_payload_bytes': tensor_payload_bytes(full_state),
        'shell_tensor_payload_bytes': tensor_payload_bytes(shell_state),
        'compiler_block_tensor_payload_bytes': [tensor_payload_bytes(s) for s in block_states],
        'compiler_total_tensor_payload_bytes': sum(tensor_payload_bytes(s) for s in block_states),
        'serialized_bytes': {
            'full': full_path.stat().st_size,
            'shell': shell_path.stat().st_size,
            'blocks': [p.stat().st_size for p in block_paths],
        },
        'source_compact_test_metrics': historical,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def new_full_model() -> g.DecoderLM:
    m = g.DecoderLM(depth=0, d=24, heads=4, mlp=24)
    m.compiler = g.GenericCompiler(depth=2, mlp=24)
    return m


def new_shell() -> g.DecoderLM:
    return g.DecoderLM(depth=0, d=24, heads=4, mlp=24)


def load_full(artifact_dir: Path) -> g.DecoderLM:
    m = new_full_model()
    state = torch.load(artifact_dir / 'compact_full.pt', map_location='cpu', weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


def load_shell(artifact_dir: Path) -> g.DecoderLM:
    m = new_shell()
    state = torch.load(artifact_dir / 'shell.pt', map_location='cpu', weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def streamed_logits(shell: g.DecoderLM, reusable: g.CausalBlock, artifact_dir: Path, tokens: torch.Tensor) -> torch.Tensor:
    h = shell.embed(tokens)
    for i in range(2):
        state = torch.load(artifact_dir / f'compiler_block{i}.pt', map_location='cpu', weights_only=True)
        reusable.load_state_dict(state)
        reusable.eval()
        del state
        h = reusable(h)
    return shell.lm_head(shell.norm(h))


def metric_accumulator():
    return {'loss_sum': 0.0, 'tokens': 0, 'correct': 0}


def add_metrics(acc: dict, logits: torch.Tensor, target: torch.Tensor) -> None:
    loss = nn.functional.cross_entropy(logits.reshape(-1, g.VOCAB), target.reshape(-1), reduction='sum')
    acc['loss_sum'] += float(loss)
    acc['tokens'] += int(target.numel())
    acc['correct'] += int((logits.argmax(-1) == target).sum())


def finish_metrics(acc: dict) -> dict:
    nll = acc['loss_sum'] / acc['tokens']
    return {'nll': nll, 'ppl': math.exp(nll), 'token_acc': acc['correct'] / acc['tokens']}


def run_s2a_equivalence(artifact_dir: Path) -> dict:
    full = load_full(artifact_dir)
    shell = load_shell(artifact_dir)
    reusable = g.CausalBlock(d=24, heads=4, mlp=24)
    reusable.eval()
    _, _, te = g.datasets()

    full_acc = metric_accumulator()
    streamed_acc = metric_accumulator()
    max_abs = 0.0
    diff_sq = 0.0
    full_sq = 0.0

    with torch.no_grad():
        for (t,) in torch.utils.data.DataLoader(te, batch_size=64, shuffle=False):
            inp = t[:, :-1]
            y = t[:, 1:]
            a = full(inp)
            b = streamed_logits(shell, reusable, artifact_dir, inp)
            d = (b - a).to(torch.float64)
            max_abs = max(max_abs, float(d.abs().max()))
            diff_sq += float((d * d).sum())
            af = a.to(torch.float64)
            full_sq += float((af * af).sum())
            add_metrics(full_acc, a, y)
            add_metrics(streamed_acc, b, y)

    fm = finish_metrics(full_acc)
    sm = finish_metrics(streamed_acc)
    rel_l2 = math.sqrt(diff_sq) / (math.sqrt(full_sq) + 1e-30)
    manifest = json.loads((artifact_dir / 'manifest.json').read_text())
    bbytes = manifest['compiler_block_tensor_payload_bytes']
    return {
        'max_absolute_logit_difference': max_abs,
        'relative_l2_logit_difference': rel_l2,
        'full_metrics': fm,
        'streamed_metrics': sm,
        'differences': {
            'nll': sm['nll'] - fm['nll'],
            'ppl': sm['ppl'] - fm['ppl'],
            'token_acc': sm['token_acc'] - fm['token_acc'],
        },
        'compiler_payload_bytes_by_construction': {
            'full_compact_resident': int(sum(bbytes)),
            'chunk_streamed_compiler': int(max(bbytes)),
            'ratio': float(max(bbytes) / sum(bbytes)),
        },
    }


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


def touch_state(state: dict) -> float:
    s = 0.0
    for v in state.values():
        if torch.is_tensor(v):
            s += float(v.sum())
    return s


def payload_probe(artifact_dir: Path, mode: str) -> dict:
    gc.collect()
    proc = psutil.Process(os.getpid())
    baseline = proc.memory_info().rss
    stop, th, samples = start_rss_sampler(proc)
    checksum = 0.0
    t0 = time.perf_counter()
    retained = []
    try:
        for i in range(LOGICAL_BLOCK_CHUNKS):
            p = artifact_dir / f'compiler_block{i % 2}.pt'
            state = torch.load(p, map_location='cpu', weights_only=True)
            checksum += touch_state(state)
            if mode == 'payload_full_resident':
                retained.append(state)
            elif mode == 'payload_streaming':
                del state
                if (i + 1) % 128 == 0:
                    gc.collect()
            else:
                raise ValueError(mode)
        elapsed = time.perf_counter() - t0
    finally:
        stop.set()
        th.join(timeout=2.0)
    peak = max(samples) if samples else proc.memory_info().rss
    manifest = json.loads((artifact_dir / 'manifest.json').read_text())
    block_bytes = manifest['compiler_block_tensor_payload_bytes']
    logical_payload = int(sum(block_bytes[i % 2] for i in range(LOGICAL_BLOCK_CHUNKS)))
    return {
        'mode': mode,
        'logical_block_chunks': LOGICAL_BLOCK_CHUNKS,
        'baseline_rss_bytes': int(baseline),
        'peak_rss_bytes': int(peak),
        'peak_rss_delta_bytes': int(max(0, peak - baseline)),
        'logical_tensor_payload_bytes': logical_payload,
        'one_chunk_max_tensor_payload_bytes': int(max(block_bytes)),
        'elapsed_seconds': elapsed,
        'checksum': checksum,
    }


def full_probe(artifact_dir: Path, mode: str) -> dict:
    gc.collect()
    proc = psutil.Process(os.getpid())
    baseline = proc.memory_info().rss
    stop, th, samples = start_rss_sampler(proc)
    try:
        _, _, te = g.datasets()
        batch = next(iter(torch.utils.data.DataLoader(te, batch_size=128, shuffle=False)))[0][:, :-1]
        if mode == 'full_compact_resident':
            model = load_full(artifact_dir)
            def f():
                with torch.no_grad():
                    return model(batch)
        elif mode == 'chunk_streamed_compiler':
            shell = load_shell(artifact_dir)
            reusable = g.CausalBlock(d=24, heads=4, mlp=24)
            reusable.eval()
            def f():
                with torch.no_grad():
                    return streamed_logits(shell, reusable, artifact_dir, batch)
        else:
            raise ValueError(mode)
        for _ in range(3):
            out = f()
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            out = f()
            times.append(time.perf_counter() - t0)
        checksum = float(out.sum())
    finally:
        stop.set()
        th.join(timeout=2.0)
    peak = max(samples) if samples else proc.memory_info().rss
    return {
        'mode': mode,
        'baseline_rss_bytes': int(baseline),
        'peak_rss_bytes': int(peak),
        'peak_rss_delta_bytes': int(max(0, peak - baseline)),
        'median_batch128_seconds': float(statistics.median(times)),
        'checksum': checksum,
    }


def run_child(script: Path, artifact_dir: Path, probe_kind: str, mode: str) -> dict:
    env = dict(os.environ)
    env.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
    })
    text = subprocess.check_output([
        sys.executable, str(script), '--artifact-dir', str(artifact_dir),
        '--probe-kind', probe_kind, '--probe-mode', mode,
    ], text=True, env=env)
    return json.loads(text)


def orchestrate(artifact_dir: Path, report_path: Path) -> dict:
    env = current_env()
    if not env_matches():
        report = {
            'experiment': 'Canaria Systems S2 learned compiler streaming',
            'status': 'ENVIRONMENT_MISMATCH',
            'expected_environment': EXPECTED_ENV,
            'observed_environment': env,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        return report

    manifest = package_learned_compact(artifact_dir)
    s2a = run_s2a_equivalence(artifact_dir)
    script = Path(__file__).resolve()
    s2a_probes = {
        m: run_child(script, artifact_dir, 'inference', m)
        for m in ['full_compact_resident', 'chunk_streamed_compiler']
    }
    s2b_probes = {
        m: run_child(script, artifact_dir, 'payload', m)
        for m in ['payload_full_resident', 'payload_streaming']
    }

    s2a_checks = {
        'max_abs_lte_1e_5': s2a['max_absolute_logit_difference'] <= 1e-5,
        'relative_l2_lte_1e_6': s2a['relative_l2_logit_difference'] <= 1e-6,
        'abs_ppl_diff_lte_1e_5': abs(s2a['differences']['ppl']) <= 1e-5,
    }
    full_rss = s2b_probes['payload_full_resident']['peak_rss_delta_bytes']
    stream_rss = s2b_probes['payload_streaming']['peak_rss_delta_bytes']
    s2b_ratio = (stream_rss / full_rss) if full_rss else None
    s2b_checks = {
        'streaming_rss_ratio_lt_0_20': bool(full_rss > 0 and stream_rss < 0.20 * full_rss),
        'payload_checksums_agree': abs(s2b_probes['payload_streaming']['checksum'] - s2b_probes['payload_full_resident']['checksum']) <= 1e-6,
    }

    status = 'PASS' if all(s2a_checks.values()) and all(s2b_checks.values()) else 'FAIL'
    report = {
        'experiment': 'Canaria Systems S2 learned compiler streaming',
        'status': status,
        'evidence_class': 'systems_runtime_format_bridge',
        'scientific_claim_use': 'DO_NOT_USE_AS_COMPOSITION_GENERALIZATION_EVIDENCE',
        'environment': env,
        'manifest': manifest,
        's2a_original_learned_model': s2a,
        's2a_fresh_process_secondary': s2a_probes,
        's2a_checks': s2a_checks,
        's2b_learned_payload_amplification': {
            'probes': s2b_probes,
            'streaming_over_full_peak_rss_delta_ratio': s2b_ratio,
            'checks': s2b_checks,
            'boundary': '4096 logical chunks repeat the two already learned G7 compiler block states only to amplify memory measurement; this is not a trained 4096-block neural model.',
        },
        'interpretation_boundary': 'S2 demonstrates chunked execution and memory-loading behavior for an actually learned Canaria compiler. It does not demonstrate deployment on a particular constrained device and does not imply arbitrary models admit the same compact representation.',
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact-dir', type=Path, required=True)
    ap.add_argument('--report', type=Path, default=Path('systems_s2_report.json'))
    ap.add_argument('--probe-kind', choices=['payload', 'inference'])
    ap.add_argument('--probe-mode')
    args = ap.parse_args()
    if args.probe_kind:
        if args.probe_kind == 'payload':
            result = payload_probe(args.artifact_dir, args.probe_mode)
        else:
            result = full_probe(args.artifact_dir, args.probe_mode)
        print(json.dumps(result))
        return
    orchestrate(args.artifact_dir, args.report)


if __name__ == '__main__':
    main()
