from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


torch.set_num_threads(1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Block(nn.Module):
    def __init__(self, d=32, heads=4, mlp=64):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, dropout=0.0)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))

    def forward(self, x):
        z = self.n1(x)
        a, _ = self.attn(z, z, z, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class SmallViT(nn.Module):
    def __init__(self, depth=4, d=32, heads=4, mlp=64):
        super().__init__()
        self.patch = nn.Conv2d(1, d, 2, 2)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, 17, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads, mlp) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 10)

    def embed(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        c = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([c, x], 1) + self.pos

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h[:, 0]))


class TinyTokenRes(nn.Module):
    def __init__(self, d: int, width: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.w1 = nn.Linear(d, width, bias=False)
        self.w2 = nn.Linear(width, d, bias=False)

    def forward(self, x):
        return x + self.w2(F.gelu(self.w1(x)))


class Cluster(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules2 = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.modules2:
            x = module(x)
        return x


class ReplacedViT(nn.Module):
    def __init__(self, teacher: SmallViT, replacement: nn.Module, start: int = 1):
        super().__init__()
        self.patch = copy.deepcopy(teacher.patch)
        self.cls = nn.Parameter(teacher.cls.detach().clone(), requires_grad=False)
        self.pos = nn.Parameter(teacher.pos.detach().clone(), requires_grad=False)
        self.pre = nn.ModuleList([copy.deepcopy(b) for b in teacher.blocks[:start]])
        self.replacement = replacement
        self.post = nn.ModuleList([copy.deepcopy(b) for b in teacher.blocks[start + 2 :]])
        self.norm = copy.deepcopy(teacher.norm)
        self.head = copy.deepcopy(teacher.head)
        for module in [self.patch, self.pre, self.post, self.norm, self.head]:
            for p in module.parameters():
                p.requires_grad_(False)

    def embed(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        c = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([c, x], 1) + self.pos

    def forward(self, x):
        h = self.embed(x)
        for block in self.pre:
            h = block(h)
        h = self.replacement(h)
        for block in self.post:
            h = block(h)
        return self.head(self.norm(h[:, 0]))


def data_split():
    d = load_digits()
    X = (d.images.astype(np.float32) / 16.0)[:, None, :, :]
    y = d.target.astype(np.int64)
    idx = np.arange(len(y))
    tr, temp = train_test_split(idx, test_size=0.30, random_state=24680, stratify=y)
    va, _te = train_test_split(temp, test_size=0.50, random_state=13579, stratify=y[temp])

    def ds(ii):
        return torch.utils.data.TensorDataset(torch.from_numpy(X[ii]), torch.from_numpy(y[ii]))

    return ds(tr), ds(va)


def train_teacher(model, ds, seed, epochs=45):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for ep in range(epochs):
        gen = torch.Generator().manual_seed(seed + ep)
        dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True, generator=gen)
        model.train()
        for x, y in dl:
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def accuracy(model, ds):
    model.eval()
    correct = total = 0
    for x, y in torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False):
        correct += int((model(x).argmax(1) == y).sum())
        total += len(y)
    return correct / total


@torch.no_grad()
def collect_span(model, ds, start=1):
    model.eval()
    a0, a1, a2 = [], [], []
    for x, _ in torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False):
        h = model.embed(x)
        for j in range(start):
            h = model.blocks[j](h)
        z1 = model.blocks[start](h)
        z2 = model.blocks[start + 1](z1)
        a0.append(h.cpu())
        a1.append(z1.cpu())
        a2.append(z2.cpu())
    return torch.cat(a0), torch.cat(a1), torch.cat(a2)


def fit_map(module, Xin, Yout, updates, seed):
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        return module
    opt = torch.optim.AdamW(params, lr=8e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    n = len(Xin)
    for _ in range(updates):
        ix = torch.randint(0, n, (64,), generator=gen)
        opt.zero_grad()
        loss = F.mse_loss(module(Xin[ix]), Yout[ix])
        loss.backward()
        opt.step()
    return module


def set_trainable(cluster, indices):
    allowed = set(indices)
    for i, module in enumerate(cluster.modules2):
        req = i in allowed
        for p in module.parameters():
            p.requires_grad_(req)
    return cluster


def nmse(pred, target, denom=None):
    if denom is None:
        denom = float(((target - target.mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12
    return float(F.mse_loss(pred, target)) / denom


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def run(seed: int):
    set_seed(seed)
    tr, va = data_split()
    teacher = SmallViT()
    teacher = train_teacher(teacher, tr, seed + 50000, 45)
    teacher_val = accuracy(teacher, va)
    result = {
        "seed": seed,
        "status": "EXPLORATORY_OUTCOME",
        "test_evaluated": False,
        "teacher_val_acc": teacher_val,
        "eligible": bool(teacher_val >= 0.95),
    }
    if not result["eligible"]:
        return result

    a0t, a1t, a2t = collect_span(teacher, tr, start=1)
    a0v, _a1v, a2v = collect_span(teacher, va, start=1)
    a0fit, a1fit, a2fit = a0t[:512], a1t[:512], a2t[:512]
    teacher_denom = float(((a2v - a2v.mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12

    c1 = fit_map(TinyTokenRes(32, 32, seed + 101001), a0fit, a1fit, 600, seed + 102001)
    c2 = fit_map(TinyTokenRes(32, 32, seed + 101002), a1fit, a2fit, 600, seed + 102002)
    base_cluster = Cluster([c1, c2])

    schedules = {
        "all_frozen": [],
        "left_only": [0],
        "right_only": [1],
        "all_unfrozen": [0, 1],
    }
    stage2_seed = seed + 120000
    stage3_init_seed = seed + 130000
    stage3_fit_seed = seed + 140000
    conditions = {}

    for name, trainable in schedules.items():
        cluster = copy.deepcopy(base_cluster)
        set_trainable(cluster, trainable)
        if trainable:
            fit_map(cluster, a0fit, a2fit, 600, stage2_seed)
        set_trainable(cluster, [])
        with torch.no_grad():
            cluster_fit_target = cluster(a0fit).detach()
            cluster_val_target = cluster(a0v).detach()
        cluster_val_nmse = nmse(cluster_val_target, a2v, teacher_denom)
        cluster_net = ReplacedViT(teacher, copy.deepcopy(cluster), start=1)
        cluster_val_acc = accuracy(cluster_net, va)

        single = TinyTokenRes(32, 64, stage3_init_seed)
        single = fit_map(single, a0fit, cluster_fit_target, 600, stage3_fit_seed)
        with torch.no_grad():
            single_val = single(a0v)
        cluster_denom = float(
            ((cluster_val_target - cluster_val_target.mean(dim=(0, 1), keepdim=True)) ** 2).mean()
        ) + 1e-12
        single_net = ReplacedViT(teacher, copy.deepcopy(single), start=1)
        conditions[name] = {
            "trainable_candidate_indices_stage2": trainable,
            "cluster": {
                "nmse_vs_original_teacher": cluster_val_nmse,
                "replacement_val_acc": cluster_val_acc,
            },
            "recursive_single": {
                "nmse_vs_cluster_teacher": nmse(single_val, cluster_val_target, cluster_denom),
                "nmse_vs_original_teacher": nmse(single_val, a2v, teacher_denom),
                "replacement_val_acc": accuracy(single_net, va),
            },
        }

    direct = TinyTokenRes(32, 64, stage3_init_seed)
    direct = fit_map(direct, a0fit, a2fit, 600, stage3_fit_seed)
    with torch.no_grad():
        direct_val = direct(a0v)
    direct_net = ReplacedViT(teacher, copy.deepcopy(direct), start=1)
    direct_metrics = {
        "nmse_vs_original_teacher": nmse(direct_val, a2v, teacher_denom),
        "replacement_val_acc": accuracy(direct_net, va),
    }
    for rec in conditions.values():
        rec["recursive_single"]["excess_original_nmse_vs_direct"] = (
            rec["recursive_single"]["nmse_vs_original_teacher"]
            - direct_metrics["nmse_vs_original_teacher"]
        )

    result.update(
        {
            "cluster_params": count_params(base_cluster),
            "single_params": count_params(direct),
            "exact_parameter_match": count_params(base_cluster) == count_params(direct) == 4096,
            "conditions": conditions,
            "direct_original_single": direct_metrics,
            "cluster_order_best_to_worst": sorted(
                conditions, key=lambda k: conditions[k]["cluster"]["nmse_vs_original_teacher"]
            ),
            "recursive_order_best_to_worst": sorted(
                conditions,
                key=lambda k: conditions[k]["recursive_single"]["nmse_vs_original_teacher"],
            ),
        }
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rec = run(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
