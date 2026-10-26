from __future__ import annotations

import argparse
import copy
import json
import math
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


class PairReplacedNet(nn.Module):
    def __init__(self, teacher, k, r1=None, r2=None, comp=None):
        super().__init__()
        self.stem = copy.deepcopy(teacher.stem)
        self.blocks = copy.deepcopy(teacher.blocks)
        self.head = copy.deepcopy(teacher.head)
        self.k, self.r1, self.r2, self.comp = k, r1, r2, comp

    def forward(self, x):
        h = F.gelu(self.stem(x))
        i = 0
        while i < len(self.blocks):
            if i == self.k:
                h = self.comp(h) if self.comp is not None else self.r2(self.r1(h))
                i += 2
            else:
                h = self.blocks[i](h)
                i += 1
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
    return (
        torch.tensor(X[train_idx]),
        torch.tensor(y[train_idx], dtype=torch.long),
        torch.tensor(X[val_idx]),
        torch.tensor(y[val_idx], dtype=torch.long),
        torch.tensor(X[test_idx]),
        torch.tensor(y[test_idx], dtype=torch.long),
    )


def train_model(seed, Xt, yt, epochs=60):
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
    opt = torch.optim.AdamW(module.parameters(), lr=8e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    n, batch = len(Xin), 128
    for _ in range(updates):
        ix = torch.randint(0, n, (batch,), generator=gen)
        opt.zero_grad()
        loss = F.mse_loss(module(Xin[ix]), Yout[ix])
        loss.backward()
        opt.step()
    return module


def fit_joint(r1, r2, Xin, Yout, updates, seed):
    opt = torch.optim.AdamW(
        list(r1.parameters()) + list(r2.parameters()),
        lr=8e-3,
        weight_decay=1e-5,
    )
    gen = torch.Generator().manual_seed(seed)
    n, batch = len(Xin), 128
    for _ in range(updates):
        ix = torch.randint(0, n, (batch,), generator=gen)
        opt.zero_grad()
        loss = F.mse_loss(r2(r1(Xin[ix])), Yout[ix])
        loss.backward()
        opt.step()
    return r1, r2


def run(seed):
    Xt, yt, Xv, yv, Xte, yte = datasets()
    teacher = train_model(seed, Xt, yt)
    train_acts, val_acts = acts(teacher, Xt), acts(teacher, Xv)
    k, updates = 0, 600
    target = val_acts[k + 2]
    denom = float(((target - target.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    teacher_val = accuracy(teacher, Xv, yv)
    teacher_test = accuracy(teacher, Xte, yte)

    rows = []
    selected_sep = None
    selected_comp = None

    for h in (2, 4, 6, 8, 12, 16, 24):
        budget = 256 * h
        r1 = fit_map(
            TinyRes(64, h, seed + 410000 + h),
            train_acts[k],
            train_acts[k + 1],
            updates,
            seed + 420000 + h,
        )
        r2 = fit_map(
            TinyRes(64, h, seed + 430000 + h),
            train_acts[k + 1],
            train_acts[k + 2],
            updates,
            seed + 440000 + h,
        )
        comp = fit_map(
            TinyRes(64, 2 * h, seed + 450000 + h),
            train_acts[k],
            train_acts[k + 2],
            updates,
            seed + 460000 + h,
        )

        sepnet = PairReplacedNet(teacher, k, r1=r1, r2=r2)
        compnet = PairReplacedNet(teacher, k, comp=comp)
        with torch.no_grad():
            sep_nmse = float(F.mse_loss(r2(r1(val_acts[k])), target)) / denom
            comp_nmse = float(F.mse_loss(comp(val_acts[k]), target)) / denom
        sep_val = accuracy(sepnet, Xv, yv)
        comp_val = accuracy(compnet, Xv, yv)
        row = {
            "h": h,
            "budget": budget,
            "sep_nmse": sep_nmse,
            "comp_nmse": comp_nmse,
            "sep_val_acc": sep_val,
            "comp_val_acc": comp_val,
            "sep_pass": bool(sep_nmse <= 0.08 and sep_val >= teacher_val - 0.02),
            "comp_pass": bool(comp_nmse <= 0.08 and comp_val >= teacher_val - 0.02),
        }
        if row["sep_pass"] and selected_sep is None:
            row["sep_test_acc"] = accuracy(sepnet, Xte, yte)
            selected_sep = dict(row)
        if row["comp_pass"] and selected_comp is None:
            row["comp_test_acc"] = accuracy(compnet, Xte, yte)
            selected_comp = dict(row)
        rows.append(row)

    j1 = TinyRes(64, 8, seed + 470008)
    j2 = TinyRes(64, 8, seed + 480008)
    j1, j2 = fit_joint(j1, j2, train_acts[k], train_acts[k + 2], 600, seed + 490008)
    with torch.no_grad():
        joint_nmse = float(F.mse_loss(j2(j1(val_acts[k])), target)) / denom
    jointnet = PairReplacedNet(teacher, k, r1=j1, r2=j2)

    result = {
        "seed": seed,
        "teacher_val_acc": teacher_val,
        "teacher_test_acc": teacher_test,
        "grid": rows,
        "selected_sep": selected_sep,
        "selected_comp": selected_comp,
        "joint_factorized_2048": {
            "nmse": joint_nmse,
            "val_acc": accuracy(jointnet, Xv, yv),
            "test_acc": accuracy(jointnet, Xte, yte),
        },
    }
    if selected_sep and selected_comp:
        result["log2_budget_ratio"] = math.log2(
            selected_comp["budget"] / selected_sep["budget"]
        )
        result["test_acc_diff_comp_minus_sep"] = (
            selected_comp["comp_test_acc"] - selected_sep["sep_test_acc"]
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "seed": args.seed,
                "sep_budget": result["selected_sep"]["budget"],
                "comp_budget": result["selected_comp"]["budget"],
                "log2_ratio": result["log2_budget_ratio"],
                "test_acc_diff": result["test_acc_diff_comp_minus_sep"],
            }
        )
    )
