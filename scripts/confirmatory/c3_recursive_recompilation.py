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
    def __init__(self, seed, d=64, depth=4):
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
        self.modules3 = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.modules3:
            x = module(x)
        return x


class TripleReplacedNet(nn.Module):
    def __init__(self, teacher, replacement):
        super().__init__()
        self.stem = copy.deepcopy(teacher.stem)
        self.block3 = copy.deepcopy(teacher.blocks[3])
        self.head = copy.deepcopy(teacher.head)
        self.replacement = replacement
        for module in (self.stem, self.block3, self.head):
            for p in module.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        h = F.gelu(self.stem(x))
        h = self.replacement(h)
        h = self.block3(h)
        return self.head(h)


def split_data():
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
    return (
        torch.tensor(X[train_idx]),
        torch.tensor(y[train_idx], dtype=torch.long),
        torch.tensor(X[val_idx]),
        torch.tensor(y[val_idx], dtype=torch.long),
        test_idx.tolist(),
    )


def materialize_test(test_idx):
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    ix = np.asarray(test_idx, dtype=np.int64)
    return torch.tensor(X[ix]), torch.tensor(y[ix], dtype=torch.long)


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


def acts(model, X):
    with torch.no_grad():
        return model(X, return_acts=True)[1]


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


def set_trainable(cluster, indices):
    allowed = set(indices)
    for i, module in enumerate(cluster.modules3):
        req = i in allowed
        for p in module.parameters():
            p.requires_grad_(req)
    return cluster


def nmse(pred, target, denom=None):
    if denom is None:
        denom = float(((target - target.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    return float(F.mse_loss(pred, target)) / denom


def run(seed):
    Xt, yt, Xv, yv, test_idx = split_data()
    teacher = train_teacher(seed, Xt, yt)
    train_acts = acts(teacher, Xt)
    val_acts = acts(teacher, Xv)
    a0t, a1t, a2t, a3t = train_acts[0], train_acts[1], train_acts[2], train_acts[3]
    a0v, a3v = val_acts[0], val_acts[3]
    teacher_denom = float(((a3v - a3v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    local = [
        fit_map(TinyRes(64, 8, seed + 101001), a0t, a1t, 600, seed + 102001),
        fit_map(TinyRes(64, 8, seed + 101002), a1t, a2t, 600, seed + 102002),
        fit_map(TinyRes(64, 8, seed + 101003), a2t, a3t, 600, seed + 102003),
    ]
    base_cluster = Cluster(local)

    schedules = {
        "all_frozen_recursive": [],
        "edges_only_recursive": [0, 2],
        "all_unfrozen_recursive": [0, 1, 2],
    }
    stage2_seed = seed + 120000
    recompile_init_seed = seed + 130000
    recompile_fit_seed = seed + 140000

    records = {}
    fitted_replacements = {}

    for name, trainable in schedules.items():
        cluster = copy.deepcopy(base_cluster)
        set_trainable(cluster, trainable)
        if trainable:
            fit_map(cluster, a0t, a3t, 600, stage2_seed)
        set_trainable(cluster, [])

        with torch.no_grad():
            cluster_train_target = cluster(a0t).detach()
            cluster_val_target = cluster(a0v).detach()
            cluster_val_out = cluster(a0v)

        cluster_val_nmse = nmse(cluster_val_out, a3v, teacher_denom)
        cluster_val_acc = accuracy(TripleReplacedNet(teacher, copy.deepcopy(cluster)), Xv, yv)

        single = TinyRes(64, 24, recompile_init_seed)
        single = fit_map(single, a0t, cluster_train_target, 600, recompile_fit_seed)
        with torch.no_grad():
            single_val_out = single(a0v)
        cluster_denom = (
            float(((cluster_val_target - cluster_val_target.mean(0, keepdim=True)) ** 2).mean())
            + 1e-12
        )
        records[name] = {
            "trainable_candidate_indices_stage2": trainable,
            "cluster_validation": {
                "nmse_vs_original_teacher": cluster_val_nmse,
                "replacement_val_acc": cluster_val_acc,
            },
            "recursive_single_validation": {
                "nmse_vs_cluster_teacher": nmse(single_val_out, cluster_val_target, cluster_denom),
                "nmse_vs_original_teacher": nmse(single_val_out, a3v, teacher_denom),
                "replacement_val_acc": accuracy(
                    TripleReplacedNet(teacher, copy.deepcopy(single)), Xv, yv
                ),
            },
        }
        fitted_replacements[name] = single

    direct = TinyRes(64, 24, recompile_init_seed)
    direct = fit_map(direct, a0t, a3t, 600, recompile_fit_seed)
    with torch.no_grad():
        direct_val_out = direct(a0v)
    direct_val = {
        "nmse_vs_original_teacher": nmse(direct_val_out, a3v, teacher_denom),
        "replacement_val_acc": accuracy(TripleReplacedNet(teacher, copy.deepcopy(direct)), Xv, yv),
    }

    # Validation metrics for every locked condition have now been recorded. Only now
    # materialize and evaluate the held-out test split. Test does not select anything.
    Xte, yte = materialize_test(test_idx)
    test_metrics = {
        name: {
            "replacement_test_acc": accuracy(
                TripleReplacedNet(teacher, copy.deepcopy(single)), Xte, yte
            )
        }
        for name, single in fitted_replacements.items()
    }
    direct_test_acc = accuracy(TripleReplacedNet(teacher, copy.deepcopy(direct)), Xte, yte)
    teacher_test_acc = accuracy(teacher, Xte, yte)

    for name in records:
        records[name]["test"] = test_metrics[name]

    au_nmse = records["all_unfrozen_recursive"]["recursive_single_validation"][
        "nmse_vs_original_teacher"
    ]
    frozen_nmse = records["all_frozen_recursive"]["recursive_single_validation"][
        "nmse_vs_original_teacher"
    ]
    edges_nmse = records["edges_only_recursive"]["recursive_single_validation"][
        "nmse_vs_original_teacher"
    ]

    return {
        "seed": seed,
        "status": "CONFIRMATORY_SEED_OUTCOME",
        "test_evaluated": True,
        "test_evaluated_only_after_all_validation_recorded": True,
        "teacher_val_acc": accuracy(teacher, Xv, yv),
        "teacher_test_acc": teacher_test_acc,
        "cluster_params": sum(p.numel() for p in base_cluster.parameters()),
        "single_params": sum(p.numel() for p in direct.parameters()),
        "records": records,
        "direct_original_single": {
            "validation": direct_val,
            "test": {"replacement_test_acc": direct_test_acc},
        },
        "paired_values": {
            "D_frozen": au_nmse - frozen_nmse,
            "R_recursive_over_direct": au_nmse / direct_val["nmse_vs_original_teacher"],
            "D_edges": au_nmse - edges_nmse,
            "test_acc_diff_all_unfrozen_minus_direct": (
                records["all_unfrozen_recursive"]["test"]["replacement_test_acc"]
                - direct_test_acc
            ),
        },
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
