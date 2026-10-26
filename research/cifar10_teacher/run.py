"""PROVISIONAL: explicit invocation only, no downloads and no official test decoding."""
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '4'
import argparse
import datetime as dt
import hashlib
import fcntl
import io
import json
import pathlib
import pickle
import platform
import random
import signal
import sys
import tarfile
import time
import numpy as np
import torch
from torch.nn import functional as F
from .model import ResNet20

SEEDS = (26102601, 26102602, 26102603)
PINNED = {'python': '3.13.5', 'torch': '2.10.0+cpu', 'numpy': '2.3.5'}
DEADLINE = dt.datetime(2026, 10, 29, 6, 50, tzinfo=dt.timezone.utc).timestamp()
SPEC = dict(epochs=120, batch_size=128, momentum=.9, weight_decay=5e-4,
            learning_rates=[[0, .1], [60, .01], [90, .001]], split_seed=20261026,
            architecture='9154339:ResNet20-option-A', dtype='float32', device='cpu', threads=4)
MODEL_CONFIG = dict(name='ResNet20', depth=20, input_channels=3, classes=10,
                    stem_channels=16, stage_channels=[16,32,64], blocks_per_stage=3,
                    shortcut='option-A', parameters=269722, initialization_source='9154339',
                    convolution=dict(kernel=3, padding=1, bias=False, stage_entry_strides=[1,2,2]),
                    batchnorm=dict(eps=1e-5, momentum=.1, affine=True, track_running_stats=True),
                    pooling='global spatial mean', classifier=dict(in_features=64, out_features=10, bias=True))
STOP = False

def canonical_hash(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())

def evaluator_identity(splits, mean, std):
    return dict(schema=1, model_config=MODEL_CONFIG, model_config_sha256=canonical_hash(MODEL_CONFIG),
                split_seed=20261026, split_indices_encoding='little-endian int64 C-order raw bytes',
                splits={name: dict(n=len(ids), sha256=sha(np.asarray(ids, dtype='<i8').tobytes()))
                        for name, ids in splits.items()},
                normalization=dict(mean=np.asarray(mean).tolist(), std=np.asarray(std).tolist(),
                                   source='teachertrain only', encoding='mean then std, little-endian float64 raw bytes',
                                   sha256=sha(np.asarray([mean, std], dtype='<f8').tobytes())))

def sha(data):
    return hashlib.sha256(data).hexdigest()

def configure(seed):
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def environment():
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                source_sha256=sha(pathlib.Path(__file__).read_bytes()+pathlib.Path(__file__).with_name('model.py').read_bytes()),
                machine=platform.machine(), platform=platform.platform(), threads=torch.get_num_threads(),
                interop=torch.get_num_interop_threads(), deterministic=torch.are_deterministic_algorithms_enabled(),
                thread_env={k: os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})

def partition(y, per_class=4000, holdout=500):
    rng = np.random.Generator(np.random.PCG64(20261026))
    parts = [[], [], []]
    for c in range(10):
        ids = rng.permutation(np.flatnonzero(y == c))
        if len(ids) != per_class + 2 * holdout:
            raise ValueError('Unexpected class count')
        for out, sub in zip(parts, (ids[:per_class], ids[per_class:per_class+holdout], ids[per_class+holdout:])):
            out.extend(sub)
    return {k: np.array(v, dtype=np.int64) for k, v in zip(('teachertrain','calibration','validation'), parts)}

def normalization(x, ids):
    s = np.zeros(3); ss = np.zeros(3); n = 0
    for j in range(0, len(ids), 256):
        a = x[ids[j:j+256]].astype(np.float64)/255
        s += a.sum((0,2,3)); ss += (a*a).sum((0,2,3)); n += a.shape[0]*1024
    mean = s/n; std = np.sqrt(ss/n-mean**2)
    if not np.all(std > 0):
        raise ValueError('Degenerate normalization')
    return mean, std

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
                   ('numpy','ndarray'):np.ndarray, ('numpy','dtype'):np.dtype}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Forbidden global {module}.{name}')
        return allowed[module, name]

def load_data(preflight):
    """Read-only access to completed preflight; opens only five training members."""
    manifest = json.loads((preflight/'data_manifest.json').read_text())
    metrics = json.loads((preflight/'metrics.json').read_text())
    if metrics.get('status') != 'complete' or not metrics.get('finite_checks_passed'):
        raise ValueError('Preflight not complete')
    archive = preflight/'cifar-10-python.tar.gz'
    h = hashlib.md5()
    with archive.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    if h.hexdigest() != 'c58f30108f718f92721af3b95e74349a':
        raise ValueError('Archive checksum mismatch')
    xx = []; yy = []
    with tarfile.open(archive, 'r:gz') as tf:
        members = tf.getmembers()
        allowed = {'cifar-10-batches-py','cifar-10-batches-py/batches.meta','cifar-10-batches-py/readme.html','cifar-10-batches-py/test_batch'} | {f'cifar-10-batches-py/data_batch_{i}' for i in range(1,6)}
        if len({m.name for m in members}) != len(members): raise ValueError('Duplicate tar members')
        for m in members:
            if not (m.name.rstrip('/') in allowed and (m.isfile() or m.isdir()) and 0 <= m.size < 40000000):
                raise ValueError('Unsafe archive member')
        for i in range(1,6):
            name = f'cifar-10-batches-py/data_batch_{i}'
            blob = tf.extractfile(name).read()
            if sha(blob) != manifest['train_member_sha256'][name]: raise ValueError('Training member mismatch')
            d = SafeUnpickler(io.BytesIO(blob), encoding='bytes').load()
            x = d[b'data']; y = np.asarray(d[b'labels'])
            if not (type(x) is np.ndarray and x.dtype == np.uint8 and x.shape == (10000,3072)
                    and y.shape == (10000,) and y.dtype.kind in 'iu' and y.min() >= 0 and y.max() <= 9):
                raise ValueError('Malformed training data')
            xx.append(x.reshape(-1,3,32,32)); yy.append(y)
    x = np.concatenate(xx); y = np.concatenate(yy)
    splits = partition(y)
    for name, ids in splits.items():
        saved = np.load(preflight/f'{name}_indices.npy', allow_pickle=False)
        if not np.array_equal(ids, saved) or sha(ids.tobytes()) != manifest['splits'][name]['index_bytes_sha256']:
            raise ValueError('Split mismatch')
    mean, std = normalization(x, splits['teachertrain'])
    if not (np.array_equal(mean, manifest['normalization_mean']) and np.array_equal(std, manifest['normalization_std'])):
        raise ValueError('Normalization mismatch')
    return x, y, splits, mean, std, sha((preflight/'data_manifest.json').read_bytes())

def augment(x, rng):
    a = np.pad(x, ((0,0),(0,0),(4,4),(4,4)))
    crops = np.empty_like(x); offsets = rng.integers(0,9,(len(x),2)); flips = rng.random(len(x)) < .5
    for k, (r,c) in enumerate(offsets):
        crop = a[k,:,r:r+32,c:c+32]; crops[k] = crop[:,:,::-1] if flips[k] else crop
    return crops

class Trainer:
    def __init__(self, x, y, train, mean, std, seed, binding, identity=None):
        configure(seed)
        self.x, self.y, self.train = x, y, train
        self.mean = torch.tensor(mean, dtype=torch.float32)[None,:,None,None]
        self.std = torch.tensor(std, dtype=torch.float32)[None,:,None,None]
        self.model = ResNet20()
        self.opt = torch.optim.SGD(self.model.parameters(), lr=.1, momentum=.9, weight_decay=5e-4)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.epoch = 0; self.cursor = 0; self.order = self.rng.permutation(train)
        self.started = time.time(); self.seed = seed; self.binding = binding
        self.env = environment()
        self.identity = identity or evaluator_identity({'teachertrain': train}, mean, std)

    def step(self, batch_size=128):
        if self.epoch >= 120: raise ValueError('Training already complete')
        self.model.train()
        lr = .1 if self.epoch < 60 else (.01 if self.epoch < 90 else .001)
        for group in self.opt.param_groups: group['lr'] = lr
        ids = self.order[self.cursor:self.cursor+batch_size]
        if not len(ids): raise ValueError('Empty batch')
        a = augment(self.x[ids], self.rng)
        inputs = (torch.from_numpy(a).float()/255-self.mean)/self.std
        targets = torch.from_numpy(self.y[ids]).long()
        self.opt.zero_grad(set_to_none=True)
        logits = self.model(inputs); loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise FloatingPointError('Non-finite/missing gradients')
        self.opt.step(); self.cursor += len(ids)
        completed = self.cursor == len(self.order)
        if completed:
            self.epoch += 1; self.cursor = 0
            self.order = self.rng.permutation(self.train) if self.epoch < 120 else np.array([], dtype=np.int64)
        return completed

    def state(self):
        return dict(schema=1, spec=SPEC, seed=self.seed, binding=self.binding, environment=self.env,
                    evaluator_identity=self.identity,
                    started=self.started, model=self.model.state_dict(), optimizer=self.opt.state_dict(),
                    epoch=self.epoch, cursor=self.cursor, order=self.order, train=self.train,
                    mean=self.mean, std=self.std, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                    augmentation_shuffle_rng=self.rng.bit_generator.state, torch_rng=torch.get_rng_state())

    def save(self, path):
        buf = io.BytesIO(); torch.save(self.state(), buf); payload = buf.getvalue()
        # One atomic envelope avoids checkpoint/sidecar torn-pair races.
        blob = b'CANARIA1\n'+sha(payload).encode()+b'\n'+payload
        tmp = path.with_suffix(path.suffix+'.tmp')
        with tmp.open('wb') as f: f.write(blob); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)

    def restore(self, path):
        magic, digest, payload = path.read_bytes().split(b'\n', 2)
        if magic != b'CANARIA1' or sha(payload).encode() != digest: raise ValueError('Checkpoint integrity failure')
        # Trusted local checkpoint only: torch serialization includes NumPy RNG state.
        s = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
        if not (s['schema'] == 1 and s['spec'] == SPEC and s['seed'] == self.seed and s['binding'] == self.binding
                and s['environment'] == self.env and s['evaluator_identity'] == self.identity and np.array_equal(s['train'], self.train)
                and torch.equal(s['mean'], self.mean) and torch.equal(s['std'], self.std)):
            raise ValueError('Checkpoint protocol/data/environment mismatch')
        epoch, cursor, order = s['epoch'], s['cursor'], s['order']
        if not (0 <= epoch <= 120 and (epoch != 120 or (cursor == 0 and len(order) == 0))
                and (epoch == 120 or (0 <= cursor < len(order) and np.array_equal(np.sort(order), np.sort(self.train))))):
            raise ValueError('Invalid dataloader cursor/order')
        self.model.load_state_dict(s['model']); self.opt.load_state_dict(s['optimizer'])
        self.epoch, self.cursor, self.order, self.started = epoch, cursor, order, s['started']
        random.setstate(s['python_rng']); np.random.set_state(s['numpy_rng'])
        self.rng.bit_generator.state = s['augmentation_shuffle_rng']; torch.set_rng_state(s['torch_rng'])

    def validate_final(self, ids, check=lambda: None):
        if self.epoch != 120: raise ValueError('Validation permitted only after epoch 120')
        if np.intersect1d(ids, self.train).size: raise ValueError('Validation/train overlap')
        self.model.eval(); correct = 0
        with torch.no_grad():
            for j in range(0, len(ids), 128):
                check(); sub = ids[j:j+128]
                a = (torch.from_numpy(self.x[sub]).float()/255-self.mean)/self.std
                logits = self.model(a)
                if not torch.isfinite(logits).all(): raise FloatingPointError('Nonfinite validation logits')
                correct += int((logits.argmax(1) == torch.from_numpy(self.y[sub])).sum())
        return dict(correct=correct, n=len(ids), accuracy=correct/len(ids), adequate=correct/len(ids) >= .85)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-provisional-plan', action='store_true', help='Parent must announce/finalize plan before use')
    p.add_argument('--preflight-artifacts', type=pathlib.Path, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if not args.execute_provisional_plan: p.error('Execution gate required; preparation does not authorize training')
    actual = dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__)
    if actual != PINNED: raise RuntimeError(f'Pinned runtime required: {PINNED}; actual: {actual}')
    if time.time() >= DEADLINE: raise TimeoutError('Research deadline reached')
    src, out = args.preflight_artifacts.resolve(), args.output.resolve()
    if out == src or src in out.parents or out in src.parents: raise ValueError('Output must be separate from preflight artifacts')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'.runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.resume and any(p.name != '.runner.lock' for p in out.iterdir()): raise ValueError('Fresh output must be empty')
    launch = time.time()
    # Kernel-delivered default SIGALRM terminates even during a long native op.
    # Never checkpoint a potentially half-applied update on hard expiry.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, max(.001, min(DEADLINE, launch+10*3600)-time.time()))
    x, y, splits, mean, std, binding = load_data(src)
    trainer = Trainer(x, y, splits['teachertrain'], mean, std, args.seed, binding, evaluator_identity(splits, mean, std))
    trainer.started = launch
    checkpoint = out/'latest.ckpt'
    if args.resume: trainer.restore(checkpoint)
    signal.setitimer(signal.ITIMER_REAL, max(.001, min(DEADLINE, trainer.started+10*3600)-time.time()))
    def check():
        if STOP: raise InterruptedError('Signal requested checkpoint and pause')
        if time.time() >= min(DEADLINE, trainer.started+10*3600): raise TimeoutError('Per-seed/deadline cap reached')
    def request_stop(*_):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    trainer.save(checkpoint)
    last_saved = time.monotonic()
    try:
        while trainer.epoch < 120:
            check(); completed = trainer.step()
            if completed or time.monotonic()-last_saved >= 300:
                trainer.save(checkpoint); last_saved = time.monotonic()
                with (out/'events.jsonl').open('a') as f:
                    f.write(json.dumps(dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(), epoch=trainer.epoch,
                                             cursor=trainer.cursor, event='checkpoint'))+'\n')
        trainer.save(out/'final_epoch120.ckpt')
        teacher_export = dict(trainer.identity, seed=args.seed, epoch=120, checkpoint='final_epoch120.ckpt',
                              checkpoint_format='CANARIA1 newline SHA256 newline torch.save payload',
                              checkpoint_sha256=sha((out/'final_epoch120.ckpt').read_bytes()),
                              data_manifest_sha256=binding, spec=SPEC, environment=trainer.env,
                              official_test_decoded=False, calibration_used=False)
        (out/'teacher_export.json.tmp').write_text(json.dumps(teacher_export, indent=2))
        os.replace(out/'teacher_export.json.tmp', out/'teacher_export.json')
        result = trainer.validate_final(splits['validation'], check)
        result.update(status='complete', seed=args.seed, epoch=120, selection='fixed final checkpoint; no early stopping',
                      test_batch_decoded=False, calibration_used=False, data_manifest_sha256=binding,
                      checkpoint_sha256=sha((out/'final_epoch120.ckpt').read_bytes()), environment=trainer.env,
                      elapsed_seconds=time.time()-trainer.started)
        (out/'metrics.json.tmp').write_text(json.dumps(result, indent=2))
        os.replace(out/'metrics.json.tmp', out/'metrics.json')
        print(json.dumps(result))
        signal.setitimer(signal.ITIMER_REAL, 0)
    except (TimeoutError, InterruptedError):
        trainer.save(checkpoint)
        raise

if __name__ == '__main__': main()
