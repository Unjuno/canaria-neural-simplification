"""In-memory fixed direct-Comp math; no data, labels, model payloads or I/O.

D is not distribution matching. Enlarging every winning-class gap can have zero
D loss while changing every soft probability. Zero loss preserves the teacher's
first-index class in exact arithmetic; numerical underflow/nonfinite quantities
are rejected, never repaired with temperature, floors, clipping or loss mixtures.

Class-preservation proof (exact arithmetic): every positive q_j times squared
violation must be zero, hence each gS_j >= gT_j >= 0. For any rival j < c,
gT_j > 0 because c is the teacher's first maximal index; therefore gS_j > 0.
For j > c, gS_j >= 0 permits only later-index ties. Student first-index argmax
is consequently c. This says nothing about matching probabilities, feature
fidelity, unseen images, or correctness of the teacher's class.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from research.cifar10_replacements.core import EndpointStats, make_arms
from research.cifar10_utility_gap.core import (
    DifferentiableFrozenTeacher, finite, make_optimizer, schedule,
    state_dict_hash, state_hash,
)
from . import protocol as p

TARGET_KEYS = frozenset(('top', 'rivals', 'probabilities', 'gaps'))


def make_cells(seed):
    """Reconstruct Sep-first, Comp-second original RNG order, then deep-copy."""
    if type(seed) is not int or seed not in p.SEEDS:
        raise ValueError('Only the three fixed teacher seeds')
    # make_arms uses fork_rng and creates Sep(rank8,rank8) BEFORE Comp(rank16).
    initial = make_arms(seed)['Comp-joint']
    return {arm: copy.deepcopy(initial) for arm in p.CELLS}


def counts(model):
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if (len(convolutions) != 2 or any(module.bias is not None for module in convolutions)
            or convolutions[0].weight.shape != (16, 32, 3, 3)
            or convolutions[1].weight.shape != (32, 16, 1, 1)
            or any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules())):
        raise ValueError('Exactly one bias-free rank16 direct-Comp map required')
    weights = sum(parameter.numel() for parameter in model.parameters())
    if weights != 5120:
        raise ValueError('Direct-Comp must contain exactly 5120 weights')
    return dict(weights=weights, nominal_convolution_macs_per_image=weights*16*16,
                output_shape=[32, 16, 16], residual_additions=1, relus=2)


def _logits(logits):
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 10
            or len(logits) == 0 or logits.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Nonempty N-by-10 float32/float64 logits required')
    finite(logits)
    return logits


def validate_targets(targets):
    """Reject edited/attached/partial or invalid target banks, including underflow."""
    if not isinstance(targets, dict) or set(targets) != TARGET_KEYS:
        raise ValueError('Exact detached teacher target fields required')
    top = targets['top']
    if (not isinstance(top, torch.Tensor) or top.ndim != 1 or not len(top)
            or top.dtype != torch.int64 or top.requires_grad
            or ((top < 0) | (top >= 10)).any()):
        raise ValueError('First-index teacher classes must be nonempty int64')
    n = len(top)
    for name in ('rivals', 'probabilities', 'gaps'):
        value = targets[name]
        dtype = torch.int64 if name == 'rivals' else torch.float64
        if (not isinstance(value, torch.Tensor) or value.shape != (n, 9)
                or value.dtype != dtype or value.device != top.device or value.requires_grad):
            raise ValueError('Exactly all nine detached same-device float64 rivals required')
        finite(value)
    all_classes = torch.arange(10, device=top.device).expand(n, -1)
    expected_rivals = all_classes[all_classes != top[:, None]].reshape(n, 9)
    if not torch.equal(targets['rivals'], expected_rivals):
        raise ValueError('Every nonwinning class exactly once in class-index order required')
    weights, gaps = targets['probabilities'], targets['gaps']
    if (weights <= 0).any() or (weights >= 1).any() or (weights.sum(1) >= 1).any():
        raise FloatingPointError('All rival weights must be finite strictly positive probabilities')
    if (gaps < 0).any() or ((targets['rivals'] < top[:, None]) & (gaps <= 0)).any():
        raise ValueError('Teacher first-index argmax gap/tie invariant violated')
    return targets


def teacher_targets(logits):
    """All nine rivals, T=1 log-softmax, gaps and probabilities in float64."""
    teacher = _logits(logits).detach().to(torch.float64)
    log_q = F.log_softmax(teacher, dim=1)
    finite(log_q)
    q = log_q.exp()
    top = teacher.argmax(1)  # PyTorch argmax selects the first maximal index.
    classes = torch.arange(10, device=teacher.device).expand(len(teacher), -1)
    rivals = classes[classes != top[:, None]].reshape(len(teacher), 9)
    gaps = teacher.gather(1, top[:, None])-teacher.gather(1, rivals)
    targets = dict(top=top.detach().clone(), rivals=rivals.detach().clone(),
                   probabilities=q.gather(1, rivals).detach().clone(), gaps=gaps.detach().clone())
    return validate_targets(targets)


def fit_normalizer(targets):
    """One full-fit mean total rival weight; never diagnostic/batch normalized."""
    validate_targets(targets)
    if len(targets['top']) != 4000:
        raise ValueError('Normalizer requires all exactly 4000 fit teacher targets')
    mu = targets['probabilities'].sum(1).mean().detach().clone()
    if mu.ndim or not torch.isfinite(mu) or mu <= 0:
        raise FloatingPointError('Finite positive full-fit normalizer required')
    return mu


def _normalizer(mu, device):
    if isinstance(mu, torch.Tensor):
        if mu.ndim or mu.dtype != torch.float64:
            raise ValueError('Normalizer tensor must be scalar float64')
        value = mu.detach().to(device=device)
    elif type(mu) is float:
        value = torch.tensor(mu, device=device, dtype=torch.float64)
    else:
        raise ValueError('Locked scalar float64/Python-float normalizer required')
    if not torch.isfinite(value) or value <= 0:
        raise FloatingPointError('Finite positive full-fit normalizer required')
    return value


def margin_preservation_loss(student_logits, targets, mu, reduction='mean'):
    """sum_i,j q_ij relu(gT_ij-gS_ij)^2 / (batch_size * fixed_mu).

    ``none`` gives per-image (already mu-normalized) contributions so reporting
    can aggregate actual example counts, including the final 32-image batch.
    A detached teacher bank and normalizer cannot receive gradients. The cast of
    student logits to float64 retains gradients through the frozen suffix.
    """
    student = _logits(student_logits).to(torch.float64)
    validate_targets(targets)
    if len(student) != len(targets['top']) or student.device != targets['top'].device:
        raise ValueError('Matching student batch and frozen teacher target rows required')
    normalizer = _normalizer(mu, student.device)
    student_gaps = student.gather(1, targets['top'][:, None])-student.gather(1, targets['rivals'])
    finite(student_gaps)
    violations = F.relu(targets['gaps']-student_gaps)
    contributions = targets['probabilities']*violations.square()
    finite(contributions)
    if ((violations > 0) & (contributions == 0)).any():
        raise FloatingPointError('Positive margin violation underflowed; no numerical repair allowed')
    totals = contributions.sum(1)
    per_image = totals/normalizer
    finite(per_image)
    if ((totals > 0) & (per_image == 0)).any():
        raise FloatingPointError('Positive margin loss normalization underflowed')
    if reduction == 'none':
        return per_image
    if reduction != 'mean':
        raise ValueError('Only per-example or mean reductions permitted')
    result = per_image.mean()
    finite(result)
    if (per_image > 0).any() and result == 0:
        raise FloatingPointError('Positive margin loss reduction underflowed')
    return result


decision_loss = margin_preservation_loss


def objective(arm, student_endpoint, teacher_endpoint, student_logits=None, targets=None, mu=None):
    if arm == 'feature':
        finite((student_endpoint, teacher_endpoint))
        if student_endpoint.shape != teacher_endpoint.shape or student_endpoint.ndim != 4 or not len(student_endpoint):
            raise ValueError('Matching nonempty NCHW endpoints required')
        loss = F.mse_loss(student_endpoint, teacher_endpoint.detach())
        finite(loss)
        return loss
    if arm == 'decision':
        return margin_preservation_loss(student_logits, targets, mu)
    raise ValueError('No mixed, searched or adaptive objective permitted')
