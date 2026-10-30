"""Standard-library-only clock/lock bootstrap, before NumPy/PyTorch imports."""
import time
PROCESS_ENTRY = time.time()
import argparse
import datetime
import fcntl
import json
import math
import os
import pathlib
import signal

DEADLINE = datetime.datetime(2026,10,29,6,50,tzinfo=datetime.timezone.utc).timestamp()
SEEDS = (26102601,26102602,26102603)

def absolute_expiry(started, now):
    if type(started) not in (int,float) or not math.isfinite(started) or started > now:
        raise ValueError('Invalid original launch timestamp')
    expiry = min(DEADLINE, started+10*3600)
    if now >= expiry: raise TimeoutError('Original per-seed/research deadline exhausted')
    return expiry

def bootstrap(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-provisional-plan', action='store_true')
    p.add_argument('--preflight-artifacts', type=pathlib.Path, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if not args.execute_provisional_plan: p.error('Execution gate required; no preparation authorization')
    src, out = args.preflight_artifacts.resolve(), args.output.resolve()
    if out == src or src in out.parents or out in src.parents:
        raise ValueError('Output must be separate from preflight artifacts')
    # Apply global deadline even while opening the tiny launch record.
    remaining = DEADLINE-time.time()
    if remaining <= 0: raise TimeoutError('Research deadline exhausted')
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'.runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = out/'launch.json'
    if args.resume:
        meta = json.loads(record.read_text())
        if meta.get('schema') != 1 or meta.get('seed') != args.seed or meta.get('preflight_artifacts') != str(src):
            raise ValueError('Launch record mismatch')
        started = meta['started']
        expiry = absolute_expiry(started, time.time())
        if meta.get('absolute_expiry') != expiry: raise ValueError('Launch expiry mismatch')
    else:
        if any(p.name != '.runner.lock' for p in out.iterdir()): raise ValueError('Fresh output must be empty')
        started = PROCESS_ENTRY
        expiry = absolute_expiry(started, time.time())
        meta = dict(schema=1, seed=args.seed, started=started, absolute_expiry=expiry, preflight_artifacts=str(src))
        # Exclusive, durable record is never rewritten on resume.
        with record.open('x') as f:
            json.dump(meta,f); f.flush(); os.fsync(f.fileno())
        fd = os.open(out,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    # Original cap applies BEFORE any NumPy/Torch import, hashing or data decode.
    remaining = expiry-time.time()
    if remaining <= 0: raise TimeoutError('Original launch cap exhausted during bootstrap')
    signal.setitimer(signal.ITIMER_REAL, remaining)
    return dict(args=args, started=started, expiry=expiry, lock=lock)
