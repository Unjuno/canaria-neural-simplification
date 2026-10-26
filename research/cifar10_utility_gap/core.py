"""Pure in-memory differentiable fixed design. No datasets, labels or file reads."""
import copy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F
from research.cifar10_replacements.core import ResidualMap, FrozenTeacher, EndpointStats
from . import protocol as p


def state_hash(model):
    return state_dict_hash(model.state_dict())


def state_dict_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode()+b'\0'+str(tensor.dtype).encode()+str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class DifferentiableFrozenTeacher(nn.Module):
    """Teacher-only wrapper: frozen parameters/BN, input gradients retained in suffix."""
    def __init__(self, admitted_teacher):
        super().__init__()
        if not isinstance(admitted_teacher, FrozenTeacher):
            raise TypeError('Existing admitted FrozenTeacher required')
        self.teacher = admitted_teacher.requires_grad_(False).eval()
        self.expected_state_sha256 = state_hash(self.teacher)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def verify_immutable(self):
        if any(module.training for module in self.modules()):
            raise ValueError('Teacher/BN is not EVAL')
        if any(param.requires_grad or param.grad is not None for param in self.parameters()):
            raise ValueError('Teacher parameters must remain gradient-free')
        if state_hash(self.teacher) != self.expected_state_sha256:
            raise ValueError('Teacher parameter or BN buffer mutated')
        return self.expected_state_sha256

    @torch.no_grad()
    def targets(self, images):
        x, _, target = self.teacher.features(images)
        return x.detach(), target.detach(), self.suffix_logits(target).detach()

    def suffix_logits(self, endpoint):
        # DO NOT use FrozenTeacher.installed_logits: that API disables gradients.
        x = endpoint
        for block in self.teacher.model.blocks[p.DESIGN['span'][1]:]:
            x = block(x)
        return self.teacher.model.fc(x.mean((2, 3)))


def make_cells(seed):
    if seed not in p.SEEDS: raise ValueError('Unplanned seed')
    cells = {}
    for width in (8, 16):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            initial = nn.Sequential(ResidualMap(width), ResidualMap(width))
        for objective in ('mse', 'kl'):
            cells[f'w{width}_{objective}'] = copy.deepcopy(initial)
    return cells


def make_optimizer(module):
    return torch.optim.Adam(module.parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0.)


def schedule(partition_record, seed):
    p.verify_partition(partition_record)
    if seed not in p.SEEDS: raise ValueError('Unplanned seed')
    ids = partition_record['fit_ids']; rng = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(64):
        order = torch.randperm(4000, generator=rng).tolist()
        batches.extend(tuple(ids[i] for i in order[start:start+128]) for start in range(0, 4000, 128))
    return tuple(batches)


def logit_kl(student_logits, teacher_logits, reduction='mean'):
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError('Matching per-example logit matrices required')
    teacher_log = F.log_softmax(teacher_logits.detach(), dim=1)
    student_log = F.log_softmax(student_logits, dim=1)
    per_example = (teacher_log.exp()*(teacher_log-student_log)).sum(1)
    if reduction == 'none': return per_example
    if reduction == 'mean': return per_example.mean()
    raise ValueError('Only per-example or mean reductions permitted')


def objective(name, student_endpoint, teacher_endpoint, student_logits, teacher_logits):
    if name == 'mse': return F.mse_loss(student_endpoint, teacher_endpoint.detach())
    if name == 'kl': return logit_kl(student_logits, teacher_logits)
    raise ValueError('No mixed or adaptive objective permitted')


def finite(value):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all(): raise FloatingPointError('Nonfinite tensor')
    elif isinstance(value, dict):
        for v in value.values(): finite(v)
    elif isinstance(value, (tuple, list)):
        for v in value: finite(v)
    elif isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError('Nonfinite scalar')
