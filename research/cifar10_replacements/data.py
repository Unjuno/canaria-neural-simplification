"""Preparation-only approved data/admission plumbing; no automatic execution.
Pinned metadata comes from source commit cfcc082, not real artifact inspection.
"""
import hashlib
import io
import json
import pathlib
import pickle
import tarfile
from dataclasses import dataclass
import numpy as np
import torch
from . import core

APPROVED_MANIFEST = {'absolute_research_deadline': '2026-10-29T06:50:00Z',
 'authorization': 'parent announced plan 2026-10-26 09:27:38 UTC; reviewed GO '
                  'source67bd076/codee2a678c nine tests PASS',
 'batch_size': 128,
 'batches_per_epoch': 313,
 'calibration': 'never train or evaluate',
 'cap_origin': 'first launch including imports, loading, and resume downtime; '
               'immutable launch.json',
 'epochs': 120,
 'failure_policy': 'nonfinite state retains last atomic good checkpoint; no '
                   'silent recovery or tuning',
 'forbidden': ['threshold tuning',
               'seed substitution',
               'early best selection',
               'CI',
               'network download',
               'charges',
               'push',
               'merge',
               'user PC'],
 'frozen_utc': '2026-10-26T09:29:16.170989+00:00',
 'last_batch_size': 64,
 'lr_zero_based': [[0, 59, 0.1], [60, 89, 0.01], [90, 119, 0.001]],
 'maximum_parallel_workers': 2,
 'official_test': 'never open or deserialize member; archive hash/traversal '
                  'permitted',
 'optimizer': {'momentum': 0.9, 'name': 'SGD', 'weight_decay': 0.0005},
 'per_seed_cap_seconds': 36000,
 'preflight_artifacts': '/workspace/shared/canaria_cifar_preflight_20261026/research/cifar10_preflight/artifacts',
 'preflight_sha256': {'calibration_indices.npy': 'f89d7cbf9d9c0aa3eb0d07d54b93a4f1ce7aad23a3e0dfa19f8c4f9efbb7bc57',
                      'cifar-10-python.tar.gz': '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce',
                      'code_commit.txt': '1fc21cfb1e3f8213f02a29ab8ac6454a39e78172f9cfdef8d5b14324d3095f9c',
                      'data_manifest.json': '8913afbe6b3e74d96f0cc4adf9411a11ac38da9a4eed1f3de735da200aa494cf',
                      'dependencies.txt': '024f2902979ab442f5048c5f6302b6d78506c03252f2b6e450056355a280c7bf',
                      'events.jsonl': 'fe0a86bbe0f2971abb811ec2ab2160b8ccf367bb7af0d454c583ef08f1620085',
                      'metrics.json': '3c49e1b69f6b35da083c51d194bdfb150bb1411519b535aaf2cfb70b6c21165d',
                      'source_sha256.txt': '57f0ae250b6a6c577d882de3da6da3be1c58f9ecc5cd5bd9778706db9a7c12d2',
                      'stdout.log': 'fe0a86bbe0f2971abb811ec2ab2160b8ccf367bb7af0d454c583ef08f1620085',
                      'step_seconds.npy': 'c363d2a14c20cef499721a42476da0cb69e25317e977c2db10a687a65d502222',
                      'teachertrain_indices.npy': 'f9347f8a0c11b08c9f66c93274822e423136ce4ebbe280a48ffa9989499b57e8',
                      'validation_indices.npy': '4501c770b46adc4822a922ed72a91e50ae8e581f27c11d2371c501e2ff17de83'},
 'python': '/workspace/shared/canaria_r87r2_training_20261026/.venv-locked/bin/python',
 'resource_policy': 'check MemAvailable before second launch; require >=2 GiB '
                    'additional free for second worker (preflight peak831MiB); '
                    'otherwise sequential',
 'restart_policy': 'no automatic retry; inspect uncertain state and notify '
                   'parent before same-job resume; original caps unchanged',
 'seeds': [26102601, 26102602, 26102603],
 'source_commit': '67bd076ec10574dc372b86f94e1d937c8c324189',
 'source_sha256': {'research/cifar10_teacher/.gitignore': '553d888952a23e007d8de3f17e3a76af89c1cd62525beb7d8527ecb5b2bdc867',
                   'research/cifar10_teacher/PREPARATION_CHECKS.md': 'df01f5589a72aef39d65c786f840632c33c42a2569e5a68e3dfa9a004519d38c',
                   'research/cifar10_teacher/PROTOCOL.md': '6c9919c76d2ba5282fc7bbe46300a5b8cd22e236ff3a6578bd2a2ba5a54eae94',
                   'research/cifar10_teacher/launch.py': '3867ea882b2c0f13334de7063236b7dc9b265cbf8a4b7927f8b96236ec39b6b0',
                   'research/cifar10_teacher/model.py': 'f04dc83e1b56cbea2adebf060c5d49312ab6cdf2ac28d625b7681acd5aea65d6',
                   'research/cifar10_teacher/run.py': 'f33a9ebecf9fd5e7b59169c3771be324f7c5c7441d208a0dfd49986811f19fb6',
                   'research/cifar10_teacher/test_synthetic.py': '89b7a1dcbad7b9335d93f8fb59e10d40bdad150f93ad8bc5d9bbb6bb894ce26d'},
 'split_seed': 20261026,
 'status': 'frozen before first teacher launch',
 'threads_per_worker': 4,
 'validation': 'only after epoch120; fixed final checkpoint; >=0.85 adequacy '
               'per seed'}
APPROVED_SPEC = {'epochs': 120, 'batch_size': 128, 'momentum': 0.9, 'weight_decay': 0.0005, 'learning_rates': [[0, 0.1], [60, 0.01], [90, 0.001]], 'split_seed': 20261026, 'architecture': '9154339:ResNet20-option-A', 'dtype': 'float32', 'device': 'cpu', 'threads': 4}
MODEL_CONFIG = {'name': 'ResNet20', 'depth': 20, 'input_channels': 3, 'classes': 10, 'stem_channels': 16, 'stage_channels': [16, 32, 64], 'blocks_per_stage': 3, 'shortcut': 'option-A', 'parameters': 269722, 'initialization_source': '9154339', 'convolution': {'kernel': 3, 'padding': 1, 'bias': False, 'stage_entry_strides': [1, 2, 2]}, 'batchnorm': {'eps': 1e-05, 'momentum': 0.1, 'affine': True, 'track_running_stats': True}, 'pooling': 'global spatial mean', 'classifier': {'in_features': 64, 'out_features': 10, 'bias': True}}
TEACHER_SOURCE_SHA256 = '5ad585a1c61f99ed21f4e67431adea3fec8851dc132ea7d3b101953dfdecf288'
OFFICIAL_MD5 = 'c58f30108f718f92721af3b95e74349a'


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
                   ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError('Forbidden global: ' + module + '.' + name)
        return allowed[module, name]


def partition(labels):
    if labels.shape != (50000,) or not np.array_equal(np.bincount(labels), np.full(10, 5000)):
        raise ValueError('Unexpected label population')
    rng = np.random.Generator(np.random.PCG64(20261026))
    parts = [[], [], []]
    for cls in range(10):
        ids = rng.permutation(np.flatnonzero(labels == cls))
        for target, group in zip(parts, (ids[:4000], ids[4000:4500], ids[4500:])):
            target.extend(group)
    return {k: np.asarray(v, dtype=np.int64) for k, v in
            zip(('teachertrain', 'calibration', 'validation'), parts)}


def normalization(images, ids):
    sums = np.zeros(3); squares = np.zeros(3); n = 0
    for start in range(0, len(ids), 256):
        x = images[ids[start:start+256]].astype(np.float64) / 255
        sums += x.sum((0, 2, 3)); squares += (x*x).sum((0, 2, 3))
        n += len(x)*1024
    mean = sums/n; std = np.sqrt(squares/n-mean**2)
    if not (np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()):
        raise ValueError('Invalid normalization')
    return mean, std


def evaluator_identity(splits, mean, std):
    return dict(schema=1, model_config=MODEL_CONFIG, model_config_sha256=core.canonical_hash(MODEL_CONFIG),
                split_seed=20261026, split_indices_encoding='little-endian int64 C-order raw bytes',
                splits={name: dict(n=len(ids), sha256=core.sha(np.asarray(ids, dtype='<i8').tobytes()))
                        for name, ids in splits.items()},
                normalization=dict(mean=mean.tolist(), std=std.tolist(), source='teachertrain only',
                    encoding='mean then std, little-endian float64 raw bytes',
                    sha256=core.sha(np.asarray([mean, std], dtype='<f8').tobytes())))


def decode_train_members(tf, manifest):
    """Never call extractfile on test_batch; member metadata traversal is allowed."""
    train_names = [f'cifar-10-batches-py/data_batch_{i}' for i in range(1, 6)]
    allowed = set(train_names) | {'cifar-10-batches-py', 'cifar-10-batches-py/batches.meta',
                               'cifar-10-batches-py/readme.html', 'cifar-10-batches-py/test_batch'}
    members = tf.getmembers()
    if len({m.name for m in members}) != len(members):
        raise ValueError('Duplicate tar member')
    for member in members:
        if not (member.name.rstrip('/') in allowed and (member.isfile() or member.isdir())
                and 0 <= member.size < 40000000):
            raise ValueError('Unsafe archive member')
    xx = []; yy = []
    for name in train_names:
        member = tf.getmember(name)
        if not member.isfile():
            raise ValueError('Training member is not a regular file')
        blob = tf.extractfile(member).read()
        if core.sha(blob) != manifest['train_member_sha256'][name]:
            raise ValueError('Training member hash mismatch')
        record = SafeUnpickler(io.BytesIO(blob), encoding='bytes').load()
        x = record[b'data']; y = np.asarray(record[b'labels'])
        if not (type(x) is np.ndarray and x.dtype == np.uint8 and x.shape == (10000, 3072)
                and y.shape == (10000,) and y.dtype.kind in 'iu' and y.min() >= 0 and y.max() <= 9):
            raise ValueError('Malformed training member')
        xx.append(x.reshape(-1, 3, 32, 32)); yy.append(y)
    return np.concatenate(xx), np.concatenate(yy)


@dataclass(frozen=True)
class ApprovedData:
    images: np.ndarray
    labels: np.ndarray
    splits: dict
    mean: np.ndarray
    std: np.ndarray
    identity: dict
    binding: str

    def require_calibration_ids(self, ids):
        actual = np.asarray(ids)
        expected = self.splits['calibration']
        if not (actual.dtype.kind in 'iu' and actual.shape == expected.shape
                and np.array_equal(actual, expected)):
            raise ValueError('Calibration membership/order is not approved partition')

    def image_batch(self, ids, *, split):
        if split not in ('calibration', 'validation'):
            raise ValueError('Only calibration or final validation allowed')
        ids = np.asarray(ids)
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not np.isin(ids, self.splits[split]).all():
            raise ValueError('Image IDs outside approved split')
        x = torch.from_numpy(self.images[ids].copy()).float()/255
        return (x-torch.as_tensor(self.mean, dtype=torch.float32)[None,:,None,None]) / torch.as_tensor(self.std, dtype=torch.float32)[None,:,None,None]


def load_approved_data(preflight, *, execution_authorized=False):
    if execution_authorized is not True:
        raise PermissionError('Parent-announced execution plan and teacher gate required')
    root = pathlib.Path(preflight)
    pins = APPROVED_MANIFEST['preflight_sha256']
    def pinned_bytes(name):
        data = (root/name).read_bytes()
        if core.sha(data) != pins[name]:
            raise ValueError('Approved artifact hash mismatch: ' + name)
        return data
    manifest_bytes = pinned_bytes('data_manifest.json')
    manifest = json.loads(manifest_bytes)
    metrics = json.loads(pinned_bytes('metrics.json'))
    if metrics.get('status') != 'complete' or metrics.get('finite_checks_passed') is not True:
        raise ValueError('Preflight incomplete')
    names = [f'cifar-10-batches-py/data_batch_{i}' for i in range(1, 6)]
    if not (manifest.get('archive_md5') == OFFICIAL_MD5 and manifest.get('split_seed') == 20261026
            and manifest.get('test_batch_deserialized') is False and manifest.get('read_members') == names
            and manifest.get('normalization_source') == 'teachertrain only'):
        raise ValueError('Preflight manifest protocol mismatch')
    archive = root/'cifar-10-python.tar.gz'
    md5 = hashlib.md5(); digest = hashlib.sha256()
    # Open once, hash and rewind same descriptor: no replacement between checks/decode.
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            md5.update(chunk); digest.update(chunk)
        if md5.hexdigest() != OFFICIAL_MD5 or digest.hexdigest() != pins[archive.name]:
            raise ValueError('Official archive MD5/SHA256 mismatch')
        stream.seek(0)
        with tarfile.open(fileobj=stream, mode='r:gz') as tf:
            images, labels = decode_train_members(tf, manifest)
    splits = partition(labels)
    for name, ids in splits.items():
        saved = np.load(io.BytesIO(pinned_bytes(name+'_indices.npy')), allow_pickle=False)
        if not (np.array_equal(saved, ids) and saved.dtype == np.int64
                and core.sha(ids.astype('<i8').tobytes()) == manifest['splits'][name]['index_bytes_sha256']):
            raise ValueError('Reconstructed split mismatch')
    mean, std = normalization(images, splits['teachertrain'])
    if not (np.array_equal(mean, manifest['normalization_mean']) and np.array_equal(std, manifest['normalization_std'])):
        raise ValueError('Reconstructed normalization mismatch')
    identity = evaluator_identity(splits, mean, std)
    for array in (images, labels, mean, std, *splits.values()):
        array.flags.writeable = False
    return ApprovedData(images, labels, splits, mean, std, identity, core.sha(manifest_bytes))


def admit_approved_teacher(export, metrics, checkpoint_blob, data, seed):
    """Require production pins in addition to core in-memory integrity checks."""
    if data.binding != APPROVED_MANIFEST['preflight_sha256']['data_manifest.json']:
        raise ValueError('Data binding not approved cfcc082 manifest')
    if data.identity.get('model_config') != MODEL_CONFIG or export.get('spec') != APPROVED_SPEC:
        raise ValueError('Unapproved teacher architecture/training recipe')
    env = export.get('environment', {})
    required = dict(python='3.13.5', torch='2.10.0+cpu', numpy='2.3.5', source_sha256=TEACHER_SOURCE_SHA256,
                    threads=4, interop=1, deterministic=True,
                    thread_env={k:'4' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
    if any(env.get(k) != v for k,v in required.items()):
        raise ValueError('Unapproved teacher source/runtime')
    if export.get('checkpoint') != 'final_epoch120.ckpt':
        raise ValueError('Only fixed-final teacher checkpoint admitted')
    teacher = core.admit_teacher(export, metrics, checkpoint_blob, data.identity, data.binding, seed)
    # Serialized normalization must agree with independently reconstructed values.
    _, _, payload = checkpoint_blob.split(b'\n', 2)
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    for key, expected in (('mean', data.mean), ('std', data.std)):
        wanted = torch.as_tensor(np.array(expected), dtype=torch.float32)[None,:,None,None]
        if not isinstance(state.get(key), torch.Tensor) or not torch.equal(state[key], wanted):
            raise ValueError('Checkpoint normalization tensor mismatch')
    if state.get('train') != data.splits['teachertrain'].tolist():
        raise ValueError('Checkpoint actual training membership mismatch')
    return teacher


@dataclass(frozen=True)
class AdmissionBundle:
    teacher: object
    seed: int
    checkpoint_sha256: str
    data_binding: str
    identity_sha256: str
    teacher_validation_correct: int
    model_state_sha256: str


def model_state_hash(model):
    digest=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        digest.update(name.encode()+b'\0'+str(tensor.dtype).encode()+str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def admit_teacher_bundle(export,metrics,checkpoint_blob,data,seed):
    teacher=admit_approved_teacher(export,metrics,checkpoint_blob,data,seed)
    return AdmissionBundle(teacher,seed,core.sha(checkpoint_blob),data.binding,
                           core.canonical_hash(data.identity),metrics['correct'],model_state_hash(teacher.model))
