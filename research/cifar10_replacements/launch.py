"""Standard-library bootstrap: call before numerical imports or artifact loading."""
import json
import math
import os
from pathlib import Path
import signal
import time

DEADLINE = 1793256600.0  # 2026-10-29T06:50:00Z; checked by synthetic test
CAP_SECONDS = 7200


def open_launch(directory, approved_plan_sha256, resume=False, now=None):
    """Durable original cap; no approval inference from elapsed time/artifact existence."""
    now = time.time() if now is None else now
    if not isinstance(approved_plan_sha256,str) or len(approved_plan_sha256)!=64 or any(ch not in '0123456789abcdef' for ch in approved_plan_sha256):
        raise ValueError('Parent-approved plan SHA256 required')
    path = Path(directory) / 'launch.json'
    if resume:
        record = json.loads(path.read_text())
        if record.get('schema')!=1 or not math.isfinite(record['started']) or not math.isfinite(record['expiry']):
            raise ValueError('Invalid launch record')
        if record['approved_plan_sha256'] != approved_plan_sha256:
            raise ValueError('Plan binding mismatch')
        if record['expiry'] != min(record['started'] + CAP_SECONDS, DEADLINE):
            raise ValueError('Original deadline invalid')
    else:
        record = dict(schema=1, approved_plan_sha256=approved_plan_sha256,
                      started=now, expiry=min(now + CAP_SECONDS, DEADLINE))
        if now >= record['expiry']: raise TimeoutError('Deadline expired')
        # Exclusive creation: never reset an existing cap.
        with path.open('x') as f:
            json.dump(record,f,sort_keys=True); f.flush(); os.fsync(f.fileno())
        fd=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    if now >= record['expiry']: raise TimeoutError('Original cap expired')
    return record


def arm_deadline(record):
    remaining=record["expiry"]-time.time()
    if remaining<=0: raise TimeoutError("Original fit cap expired")
    signal.signal(signal.SIGALRM,signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL,remaining)


def bootstrap(directory,approved_plan_sha256,*,resume=False):
    """Reviewed future caller entry: BEFORE importing torch/numpy or runtime.

    Returns (durable record, held worker-lock file). Keep handle alive for the
    entire run. Single replacement worker;4threads. Does not authorize execution.
    """
    import sys
    import fcntl
    if 'torch' in sys.modules or 'numpy' in sys.modules:
        raise RuntimeError('Bootstrap must precede numerical imports')
    worker=open('/workspace/shared/canaria_cifar_replacement_worker.lock','a+')
    try:
        fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            os.environ[key]='4'
        record=open_launch(directory,approved_plan_sha256,resume=resume)
        arm_deadline(record)
        return record,worker
    except BaseException:
        worker.close()
        raise
