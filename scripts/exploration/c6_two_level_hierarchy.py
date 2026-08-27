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


class ResBlock(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128, d)

    def forward(self, x):
        z = self.ln(x)
        z = F.gelu(self.fc1(z))
        return x + self.fc2(z)


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
    def __init__(self, d, width, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.w1 = nn.Linear(d, width, bias=False)
        self.w2 = nn.Linear(width, d, bias=False)

    def forward(self, x):
        return x + self.w2(F.gelu(self.w1(x)))


class Cluster(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.mods = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.mods:
            x = module(x)
        return x


class WholeReplacedNet(nn.Module):
    def __init__(self, teacher, replacement):
        super().__init__()
        self.stem = copy.deepcopy(teacher.stem)
        self.replacement = replacement
        self.head = copy.deepcopy(teacher.head)
        for module in (self.stem, self.head):
            for p in module.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        h = F.gelu(self.stem(x))
        h = self.replacement(h)
        return self.head(h)


def split_data():
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    idx = np.arange(len(y))
    train_idx, _test_idx = train_test_split(
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


@torch.no_grad()
def accuracy(model, X, y):
    return float((model(X).argmax(-1) == y).float().mean())


@torch.no_grad()
def acts(model, X):
    return model(X, return_acts=True)[1]


def fit_map(module, Xin, Yout, updates, seed):
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        return module
    opt = torch.optim.AdamW(params, lr=8e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    n = len(Xin)
    for _ in range(updates):
        ix = torch.randint(0, n, (128,), generator=gen)
        opt.zero_grad()
        loss = F.mse_loss(module(Xin[ix]), Yout[ix])
        loss.backward()
        opt.step()
    return module


def set_all_trainable(module, value: bool):
    for p in module.parameters():
        p.requires_grad_(value)
    return module


def nmse(pred, target, denom=None):
    if denom is None:
        denom = float(((target - target.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    return float(F.mse_loss(pred, target)) / denom


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def final_metrics(single, immediate_train_target, immediate_val_target, a0t, a0v, a4v, original_denom, teacher, Xv, yv, fit_seed):
    single = fit_map(single, a0t, immediate_train_target, 600, fit_seed)
    with torch.no_grad():
        out = single(a0v)
    immediate_denom = float(((immediate_val_target - immediate_val_target.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    return single, {
        "nmse_vs_immediate_canaria_teacher": nmse(out, immediate_val_target, immediate_denom),
        "nmse_vs_original_teacher": nmse(out, a4v, original_denom),
        "replacement_val_acc": accuracy(WholeReplacedNet(teacher, copy.deepcopy(single)), Xv, yv),
    }


def run(seed: int):
    set_seed(seed)
    Xt, yt, Xv, yv = split_data()
    teacher = train_teacher(seed, Xt, yt)
    teacher_val_acc = accuracy(teacher, Xv, yv)
    result = {
        "seed": seed,
        "status": "EXPLORATORY_OUTCOME",
        "teacher_val_acc": teacher_val_acc,
        "eligible": bool(teacher_val_acc >= 0.95),
        "test_evaluated": False,
    }
    if not result["eligible"]:
        return result

    at = acts(teacher, Xt)
    av = acts(teacher, Xv)
    a0t, a1t, a2t, a3t, a4t = at
    a0v, a1v, a2v, a3v, a4v = av
    original_denom = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    locals4 = [
        fit_map(TinyRes(64, 8, seed + 101001), a0t, a1t, 600, seed + 102001),
        fit_map(TinyRes(64, 8, seed + 101002), a1t, a2t, 600, seed + 102002),
        fit_map(TinyRes(64, 8, seed + 101003), a2t, a3t, 600, seed + 102003),
        fit_map(TinyRes(64, 8, seed + 101004), a3t, a4t, 600, seed + 102004),
    ]

    pair12 = Cluster([copy.deepcopy(locals4[0]), copy.deepcopy(locals4[1])])
    pair34 = Cluster([copy.deepcopy(locals4[2]), copy.deepcopy(locals4[3])])
    set_all_trainable(pair12, True)
    set_all_trainable(pair34, True)
    pair12 = fit_map(pair12, a0t, a2t, 600, seed + 120001)
    pair34 = fit_map(pair34, a2t, a4t, 600, seed + 120002)
    set_all_trainable(pair12, False)
    set_all_trainable(pair34, False)

    with torch.no_grad():
        pair12_t = pair12(a0t).detach()
        pair12_v = pair12(a0v).detach()
        pair34_t_on_a2 = pair34(a2t).detach()
        pair34_v_on_a2 = pair34(a2v).detach()
        lower_ir_t = pair34(pair12_t).detach()
        lower_ir_v = pair34(pair12_v).detach()

    c12 = TinyRes(64, 16, seed + 130001)
    c12 = fit_map(c12, a0t, pair12_t, 600, seed + 140001)
    c34 = TinyRes(64, 16, seed + 130002)
    c34 = fit_map(c34, a2t, pair34_t_on_a2, 600, seed + 140002)

    pair12_denom = float(((pair12_v - pair12_v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    pair34_denom = float(((pair34_v_on_a2 - pair34_v_on_a2.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    with torch.no_grad():
        c12_v = c12(a0v)
        c34_v_on_a2 = c34(a2v)
        pre_t = c34(c12(a0t)).detach()
        pre_v = c34(c12_v).detach()

    lower_ir_denom = float(((lower_ir_v - lower_ir_v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    pair_units = {
        "C12_nmse_vs_pair12_teacher": nmse(c12_v, pair12_v, pair12_denom),
        "C34_nmse_vs_pair34_teacher": nmse(c34_v_on_a2, pair34_v_on_a2, pair34_denom),
    }
    lower_ir = {
        "nmse_vs_original_teacher": nmse(lower_ir_v, a4v, original_denom),
        "replacement_val_acc": accuracy(WholeReplacedNet(teacher, copy.deepcopy(Cluster([pair12, pair34]))), Xv, yv),
    }
    pre_level2 = {
        "nmse_vs_lower_ir": nmse(pre_v, lower_ir_v, lower_ir_denom),
        "nmse_vs_original_teacher": nmse(pre_v, a4v, original_denom),
        "replacement_val_acc": accuracy(WholeReplacedNet(teacher, copy.deepcopy(Cluster([c12, c34]))), Xv, yv),
    }

    level2_pair = Cluster([copy.deepcopy(c12), copy.deepcopy(c34)])
    set_all_trainable(level2_pair, True)
    level2_pair = fit_map(level2_pair, a0t, lower_ir_t, 600, seed + 150001)
    set_all_trainable(level2_pair, False)
    with torch.no_grad():
        post_t = level2_pair(a0t).detach()
        post_v = level2_pair(a0v).detach()
    post_level2 = {
        "nmse_vs_lower_ir": nmse(post_v, lower_ir_v, lower_ir_denom),
        "nmse_vs_original_teacher": nmse(post_v, a4v, original_denom),
        "replacement_val_acc": accuracy(WholeReplacedNet(teacher, copy.deepcopy(level2_pair)), Xv, yv),
    }

    init_seed = seed + 160001
    fit_seed = seed + 170001
    finals = {}
    s_no, finals["hierarchical_no_level2_adapt"] = final_metrics(
        TinyRes(64, 32, init_seed), pre_t, pre_v, a0t, a0v, a4v, original_denom, teacher, Xv, yv, fit_seed
    )
    s_joint, finals["hierarchical_level2_joint_adapt"] = final_metrics(
        TinyRes(64, 32, init_seed), post_t, post_v, a0t, a0v, a4v, original_denom, teacher, Xv, yv, fit_seed
    )
    s_flat, finals["flat_lower_ir_single"] = final_metrics(
        TinyRes(64, 32, init_seed), lower_ir_t, lower_ir_v, a0t, a0v, a4v, original_denom, teacher, Xv, yv, fit_seed
    )
    s_direct, finals["direct_original_single"] = final_metrics(
        TinyRes(64, 32, init_seed), a4t, a4v, a0t, a0v, a4v, original_denom, teacher, Xv, yv, fit_seed
    )

    direct_nmse = finals["direct_original_single"]["nmse_vs_original_teacher"]
    flat_nmse = finals["flat_lower_ir_single"]["nmse_vs_original_teacher"]
    for name in ("hierarchical_no_level2_adapt", "hierarchical_level2_joint_adapt"):
        finals[name]["ratio_vs_direct_original_nmse"] = finals[name]["nmse_vs_original_teacher"] / direct_nmse
        finals[name]["ratio_vs_flat_lower_ir_nmse"] = finals[name]["nmse_vs_original_teacher"] / flat_nmse

    result.update({
        "budget": {
            "level0_total_params": sum(count_params(m) for m in locals4),
            "level1_total_params": count_params(c12) + count_params(c34),
            "level2_params": count_params(s_joint),
            "exact_4096_match": (
                sum(count_params(m) for m in locals4) == count_params(c12) + count_params(c34) == count_params(s_joint) == 4096
            ),
        },
        "pair_units": pair_units,
        "lower_ir_full_teacher": lower_ir,
        "pre_level2_pair": pre_level2,
        "post_level2_joint_pair": post_level2,
        "finals": finals,
        "final_order_best_to_worst": sorted(
            finals, key=lambda k: finals[k]["nmse_vs_original_teacher"]
        ),
        "derived": {
            "level2_joint_improvement_nmse": finals["hierarchical_level2_joint_adapt"]["nmse_vs_original_teacher"] - finals["hierarchical_no_level2_adapt"]["nmse_vs_original_teacher"],
            "depth2_cost_vs_flat_lower_ir_nmse": finals["hierarchical_level2_joint_adapt"]["nmse_vs_original_teacher"] - flat_nmse,
            "depth2_ratio_vs_flat_lower_ir": finals["hierarchical_level2_joint_adapt"]["nmse_vs_original_teacher"] / flat_nmse,
            "depth2_ratio_vs_direct_original": finals["hierarchical_level2_joint_adapt"]["nmse_vs_original_teacher"] / direct_nmse,
        },
    })
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
