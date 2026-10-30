"""Source-only PROVISIONAL comparison primitives. No data readers or fit runner."""
import copy
import hashlib
import io
import json
import math

import torch
from torch import nn
from torch.nn import functional as F
from research.cifar10_teacher.model import ResNet20

SEEDS = (26102601, 26102602, 26102603)
SPAN = (4, 6)  # zero-based, half-open: stage2 blocks2–3
RECIPE = dict(updates=1000, batch_size=128, optimizer='Adam', lr=1e-3,
              betas=(.9, .999), eps=1e-8, weight_decay=0., schedule='constant')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


class ResidualMap(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.down = nn.Conv2d(32, rank, 3, padding=1, bias=False)
        self.up = nn.Conv2d(rank, 32, 1, bias=False)

    def forward(self, x):
        return F.relu(x + self.up(F.relu(self.down(x))))


def make_arms(seed):
    # Explicit torch default Conv2d initialization, isolated from caller RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        sep = nn.Sequential(ResidualMap(8), ResidualMap(8))
        comp = nn.Sequential(ResidualMap(16))
    return {'Comp-joint': comp, 'Sep-local': sep, 'Sep-joint': copy.deepcopy(sep)}


def make_optimizer(module):
    return torch.optim.Adam(module.parameters(), lr=RECIPE['lr'], betas=RECIPE['betas'],
                            eps=RECIPE['eps'], weight_decay=RECIPE['weight_decay'])


class FrozenTeacher(nn.Module):
    """Owns only frozen teacher; mode changes cannot enable any teacher BN."""
    def __init__(self, model):
        super().__init__()
        if not isinstance(model, ResNet20):
            raise TypeError('Expected pinned ResNet20')
        self.model = model.requires_grad_(False).eval()
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def features(self, images):
        self.eval()
        x = F.relu(self.model.bn(self.model.stem(images)))
        for block in self.model.blocks[:SPAN[0]]:
            x = block(x)
        middle = self.model.blocks[4](x)
        target = self.model.blocks[5](middle)
        return tuple(t.detach() for t in (x, middle, target))

    @torch.no_grad()
    def installed_logits(self, images, replacement):
        self.eval()
        x, _, _ = self.features(images)
        x = replacement(x)
        for block in self.model.blocks[SPAN[1]:]:
            x = block(x)
        return self.model.fc(x.mean((2, 3)))


def objective(arm_name, replacement, endpoints):
    x, middle, target = (t.detach() for t in endpoints)
    if arm_name == 'Sep-local':
        if len(replacement) != 2:
            raise ValueError('Local arm requires two maps')
        # Mean over batch/channel/space per endpoint, SUM across local modules.
        return F.mse_loss(replacement[0](x), middle) + F.mse_loss(replacement[1](middle), target)
    if arm_name not in ('Comp-joint', 'Sep-joint'):
        raise ValueError('Unknown arm')
    return F.mse_loss(replacement(x), target)


def calibration_batches(ids, seed, updates=1000, batch_size=128):
    """Same immutable ID schedule consumed by every arm; no labels or I/O."""
    if len(ids) != 5000 or len(set(ids)) != 5000:
        raise ValueError('Exactly 5000 unique admitted calibration IDs required')
    if updates != RECIPE['updates'] or batch_size != RECIPE['batch_size']:
        raise ValueError('Recipe changes require a new reviewed protocol')
    generator = torch.Generator().manual_seed(seed)
    result = []
    while len(result) < updates:
        order = torch.randperm(len(ids), generator=generator).tolist()
        for start in range(0, len(ids), batch_size):
            result.append(tuple(ids[i] for i in order[start:start+batch_size]))
            if len(result) == updates:
                break
    return tuple(result)


def uncentered_energy_nmse(squared_error_sum, teacher_squared_sum):
    """Aggregate all elements before ratio, NOT average per-batch ratios."""
    if not (math.isfinite(squared_error_sum) and math.isfinite(teacher_squared_sum)):
        raise ValueError('Nonfinite endpoint sums')
    if squared_error_sum < 0 or teacher_squared_sum <= 0:
        raise ValueError('Invalid endpoint sums; zero target energy is undefined')
    return squared_error_sum / teacher_squared_sum


def admit_teacher(export, metrics, checkpoint_blob, expected_identity, expected_binding, expected_seed):
    """In-memory verifier; caller supplies approved manifests. No path/data reads.

    Integrity/binding checks are not provenance authentication. Future caller must
    source expected identity from the independently approved preflight manifest.
    """
    required = {'schema', 'model_config', 'model_config_sha256', 'split_seed',
                'split_indices_encoding', 'splits', 'normalization'}
    if set(expected_identity) != required or expected_identity['schema'] != 1 or expected_identity['split_seed'] != 20261026:
        raise ValueError('Incomplete approved evaluator identity')
    if {k: v.get('n') for k, v in expected_identity['splits'].items()} != {'teachertrain': 40000, 'calibration': 5000, 'validation': 5000}:
        raise ValueError('Invalid approved split sizes')
    if expected_identity['normalization'].get('source') != 'teachertrain only':
        raise ValueError('Invalid normalization source')
    if expected_seed not in SEEDS:
        raise ValueError('Unplanned seed')
    for record in (export, metrics):
        if record.get('seed') != expected_seed or record.get('epoch') != 120:
            raise ValueError('Only planned fixed final epoch120 admitted')
        if record.get('data_manifest_sha256') != expected_binding:
            raise ValueError('Data identity mismatch')
        if record.get('checkpoint_sha256') != sha(checkpoint_blob):
            raise ValueError('Checkpoint full-file mismatch')
    if (metrics.get('status') != 'complete' or metrics.get('adequate') is not True
            or metrics.get('n') != 5000 or type(metrics.get('correct')) is not int
            or not 4250 <= metrics['correct'] <= 5000
            or metrics.get('accuracy') != metrics['correct'] / 5000):
        raise ValueError('Missing/failed final validation >=85% gate')
    if (export.get('official_test_decoded') is not False or metrics.get('test_batch_decoded') is not False
            or export.get('calibration_used') is not False or metrics.get('calibration_used') is not False):
        raise ValueError('Split isolation flags invalid')
    for key, value in expected_identity.items():
        if export.get(key) != value:
            raise ValueError('Evaluator identity mismatch: ' + key)
    if expected_identity.get('model_config_sha256') != canonical_hash(expected_identity.get('model_config')):
        raise ValueError('Model configuration hash invalid')
    if export.get('environment') != metrics.get('environment'):
        raise ValueError('Environment mismatch')
    try:
        magic, digest, payload = checkpoint_blob.split(b'\n', 2)
    except ValueError as exc:
        raise ValueError('Invalid checkpoint envelope') from exc
    if magic != b'CANARIA1' or digest != sha(payload).encode():
        raise ValueError('Checkpoint payload mismatch')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if not (state.get('schema') == 2 and state.get('seed') == expected_seed
            and state.get('epoch') == 120 and state.get('cursor') == 0 and state.get('order') == []
            and state.get('binding') == expected_binding and state.get('evaluator_identity') == expected_identity
            and state.get('environment') == export.get('environment') and state.get('spec') == export.get('spec')):
        raise ValueError('Checkpoint internal identity/final-state mismatch')
    model = ResNet20()
    model.load_state_dict(state['model'], strict=True)
    if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
        raise ValueError('Nonfinite teacher state')
    return FrozenTeacher(model)


class EndpointStats:
    """Primary centered NMSE: per-(C,H,W) mean across full sample N.

    Equivalent to mean squared error / (mean centered teacher variance +1e-12).
    Therefore SSE denominator gets epsilon multiplied by N*C*H*W.
    """
    def __init__(self):
        self.n=0; self.sse=0.; self.sum_y=None; self.sum_y2=None

    def update(self,predicted,target):
        if predicted.shape!=target.shape or target.ndim!=4 or target.shape[0]==0:
            raise ValueError('Expected matching nonempty NCHW endpoint tensors')
        p=predicted.detach().double().cpu(); y=target.detach().double().cpu()
        if not torch.isfinite(p).all() or not torch.isfinite(y).all():
            raise ValueError('Nonfinite endpoints')
        if self.sum_y is not None and self.sum_y.shape!=y.shape[1:]:
            raise ValueError('Endpoint feature shape changed')
        self.n+=len(y); self.sse+=float(((p-y)**2).sum())
        sy=y.sum(0); sy2=(y*y).sum(0)
        self.sum_y=sy if self.sum_y is None else self.sum_y+sy
        self.sum_y2=sy2 if self.sum_y2 is None else self.sum_y2+sy2

    def result(self):
        if not self.n: raise ValueError('No endpoint samples')
        raw=float((self.sum_y2-self.sum_y.square()/self.n).sum())
        # Tiny cancellation is clamped; material negative variance is invalid.
        tolerance=1e-12*max(float(self.sum_y2.sum()),1.)
        if raw < -tolerance: raise ValueError('Invalid negative centered variance')
        denom=max(raw,0.)+1e-12*self.n*self.sum_y.numel()
        energy=float(self.sum_y2.sum())
        return dict(centered_span_nmse=self.sse/denom, sse=self.sse,n=self.n,
                    centered_teacher_ss=max(raw,0.),epsilon=1e-12,
                    uncentered_energy_nmse=self.sse/energy if energy>0 else None)
