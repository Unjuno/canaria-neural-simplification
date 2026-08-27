from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


torch.set_num_threads(1)


class ResBlock(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128, d)

    def forward(self, x):
        z = self.ln(x)
        z = F.gelu(self.fc1(z))
        z = self.fc2(z)
        return x + z


class Net(nn.Module):
    def __init__(self, seed, d=64, depth=7):
        super().__init__()
        torch.manual_seed(seed)
        self.stem = nn.Linear(64, d)
        self.blocks = nn.ModuleList([ResBlock(d) for _ in range(depth)])
        self.head = nn.Linear(d, 10)

    def forward(self, x, return_acts=False):
        h = F.gelu(self.stem(x))
        acts = [h]
        for block in self.blocks:
            h = block(h)
            acts.append(h)
        logits = self.head(h)
        return (logits, acts) if return_acts else logits


class TinyRes(nn.Module):
    def __init__(self, d, w, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.w1 = nn.Linear(d, w, bias=False)
        self.w2 = nn.Linear(w, d, bias=False)

    def forward(self, x):
        return x + self.w2(F.gelu(self.w1(x)))


class Cluster(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modulesN = nn.ModuleList(modules)

    def forward(self, x, return_acts=False):
        acts = [x]
        for module in self.modulesN:
            x = module(x)
            acts.append(x)
        return (x, acts) if return_acts else x


class SpanReplacedNet(nn.Module):
    def __init__(self, teacher, replacement):
        super().__init__()
        self.stem = copy.deepcopy(teacher.stem)
        self.downstream_block = copy.deepcopy(teacher.blocks[6])
        self.head = copy.deepcopy(teacher.head)
        self.replacement = replacement
        for module in (self.stem, self.downstream_block, self.head):
            for p in module.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        h = F.gelu(self.stem(x))
        h = self.replacement(h)
        h = self.downstream_block(h)
        return self.head(h)


def datasets():
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.25, random_state=1234, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.2,
        random_state=5678,
        stratify=y[train_idx],
    )
    _ = test_idx
    return (
        torch.tensor(X[train_idx]),
        torch.tensor(y[train_idx], dtype=torch.long),
        torch.tensor(X[val_idx]),
        torch.tensor(y[val_idx], dtype=torch.long),
    )


def train_teacher(seed, Xt, yt, epochs=60):
    model = Net(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 999)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), 64):
            ix = perm[i : i + 64]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[ix]), yt[ix])
            loss.backward()
            opt.step()
    return model


def accuracy(model, X, y):
    with torch.no_grad():
        return float((model(X).argmax(-1) == y).float().mean())


def teacher_acts(model, X):
    with torch.no_grad():
        return model(X, return_acts=True)[1]


def cluster_acts(cluster, X):
    with torch.no_grad():
        _, acts = cluster(X, return_acts=True)
        return [a.detach() for a in acts]


def fit_map(module, Xin, Yout, updates, seed):
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        return module
    opt = torch.optim.AdamW(params, lr=8e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    n, batch = len(Xin), 128
    for _ in range(updates):
        ix = torch.randint(0, n, (batch,), generator=gen)
        opt.zero_grad()
        loss = F.mse_loss(module(Xin[ix]), Yout[ix])
        loss.backward()
        opt.step()
    return module


def set_all_trainable(module, value):
    for p in module.parameters():
        p.requires_grad_(value)
    return module


def nmse(pred, target, denom=None):
    if denom is None:
        denom = float(((target - target.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    return float(F.mse_loss(pred, target)) / denom


def eval_replacement(teacher, replacement, Xv, yv):
    return accuracy(SpanReplacedNet(teacher, copy.deepcopy(replacement)), Xv, yv)


def run(seed):
    Xt, yt, Xv, yv = datasets()
    teacher = train_teacher(seed, Xt, yt)
    ta = teacher_acts(teacher, Xt)
    va = teacher_acts(teacher, Xv)
    a0t, a6t = ta[0], ta[6]
    a0v, a6v = va[0], va[6]
    original_denom = float(((a6v - a6v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    locals6 = []
    for i in range(6):
        locals6.append(
            fit_map(
                TinyRes(64, 4, seed + 101000 + i),
                ta[i],
                ta[i + 1],
                600,
                seed + 102000 + i,
            )
        )
    local_cluster = Cluster(locals6)

    adapted6 = copy.deepcopy(local_cluster)
    set_all_trainable(adapted6, True)
    fit_map(adapted6, a0t, a6t, 600, seed + 120000)
    set_all_trainable(adapted6, False)
    with torch.no_grad():
        adapted6_train_target = adapted6(a0t).detach()
        adapted6_val_target = adapted6(a0v).detach()
    adapted6_denom = (
        float(((adapted6_val_target - adapted6_val_target.mean(0, keepdim=True)) ** 2).mean())
        + 1e-12
    )
    adapted6_metrics = {
        "nmse_vs_original_teacher": nmse(adapted6_val_target, a6v, original_denom),
        "validation_accuracy": eval_replacement(teacher, adapted6, Xv, yv),
    }

    shared_single_init = seed + 130000
    shared_single_fit = seed + 140000

    one_level = TinyRes(64, 24, shared_single_init)
    one_level = fit_map(one_level, a0t, adapted6_train_target, 600, shared_single_fit)
    with torch.no_grad():
        one_val = one_level(a0v)
    one_metrics = {
        "nmse_vs_adapted_six_teacher": nmse(one_val, adapted6_val_target, adapted6_denom),
        "nmse_vs_original_teacher": nmse(one_val, a6v, original_denom),
        "validation_accuracy": eval_replacement(teacher, one_level, Xv, yv),
    }

    ca_train = cluster_acts(adapted6, a0t)
    ca_val = cluster_acts(adapted6, a0v)
    pair_modules = [
        fit_map(TinyRes(64, 8, seed + 151000), ca_train[0], ca_train[2], 600, seed + 152000),
        fit_map(TinyRes(64, 8, seed + 151001), ca_train[2], ca_train[4], 600, seed + 152001),
        fit_map(TinyRes(64, 8, seed + 151002), ca_train[4], ca_train[6], 600, seed + 152002),
    ]
    pair_cluster = Cluster(pair_modules)
    set_all_trainable(pair_cluster, True)
    fit_map(pair_cluster, a0t, adapted6_train_target, 600, seed + 160000)
    set_all_trainable(pair_cluster, False)
    with torch.no_grad():
        pair_train_target = pair_cluster(a0t).detach()
        pair_val_target = pair_cluster(a0v).detach()
    pair_denom = (
        float(((pair_val_target - pair_val_target.mean(0, keepdim=True)) ** 2).mean())
        + 1e-12
    )
    pair_metrics = {
        "nmse_vs_adapted_six_teacher": nmse(pair_val_target, adapted6_val_target, adapted6_denom),
        "nmse_vs_original_teacher": nmse(pair_val_target, a6v, original_denom),
        "validation_accuracy": eval_replacement(teacher, pair_cluster, Xv, yv),
    }

    two_level = TinyRes(64, 24, shared_single_init)
    two_level = fit_map(two_level, a0t, pair_train_target, 600, shared_single_fit)
    with torch.no_grad():
        two_val = two_level(a0v)
    two_metrics = {
        "nmse_vs_pair_cluster_teacher": nmse(two_val, pair_val_target, pair_denom),
        "nmse_vs_adapted_six_teacher": nmse(two_val, adapted6_val_target, adapted6_denom),
        "nmse_vs_original_teacher": nmse(two_val, a6v, original_denom),
        "validation_accuracy": eval_replacement(teacher, two_level, Xv, yv),
    }

    direct = TinyRes(64, 24, shared_single_init)
    direct = fit_map(direct, a0t, a6t, 600, shared_single_fit)
    with torch.no_grad():
        direct_val = direct(a0v)
    direct_metrics = {
        "nmse_vs_original_teacher": nmse(direct_val, a6v, original_denom),
        "validation_accuracy": eval_replacement(teacher, direct, Xv, yv),
    }

    return {
        "seed": seed,
        "status": "EXPLORATORY_OUTCOME",
        "test_evaluated": False,
        "teacher_validation_accuracy": accuracy(teacher, Xv, yv),
        "parameter_budgets": {
            "local_cluster": sum(p.numel() for p in local_cluster.parameters()),
            "adapted_six_cluster": sum(p.numel() for p in adapted6.parameters()),
            "pair_cluster": sum(p.numel() for p in pair_cluster.parameters()),
            "one_level_single": sum(p.numel() for p in one_level.parameters()),
            "two_level_single": sum(p.numel() for p in two_level.parameters()),
            "direct_single": sum(p.numel() for p in direct.parameters()),
        },
        "adapted_six_cluster": adapted6_metrics,
        "one_level_recursive_single": one_metrics,
        "pair_cluster_recursive_level1": pair_metrics,
        "two_level_recursive_single": two_metrics,
        "direct_original_single": direct_metrics,
        "ratios": {
            "two_level_over_one_level_original_nmse": (
                two_metrics["nmse_vs_original_teacher"] / one_metrics["nmse_vs_original_teacher"]
            ),
            "two_level_over_direct_original_nmse": (
                two_metrics["nmse_vs_original_teacher"] / direct_metrics["nmse_vs_original_teacher"]
            ),
            "one_level_over_direct_original_nmse": (
                one_metrics["nmse_vs_original_teacher"] / direct_metrics["nmse_vs_original_teacher"]
            ),
        },
        "recursive_fit_targets_after_adapted_six_freeze": "adapted-six internal/final outputs or pair-cluster outputs only; original a6 not used",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = run(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
