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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


torch.set_num_threads(1)


class ResBlock(nn.Module):
    def __init__(self, d: int = 64):
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
    def __init__(self, seed: int, input_dim: int, d: int = 64, depth: int = 4):
        super().__init__()
        torch.manual_seed(seed)
        self.stem = nn.Linear(input_dim, d)
        self.blocks = nn.ModuleList([ResBlock(d) for _ in range(depth)])
        self.head = nn.Linear(d, 1)

    def forward(self, x, return_acts: bool = False):
        h = F.gelu(self.stem(x))
        acts = [h]
        for block in self.blocks:
            h = block(h)
            acts.append(h)
        pred = self.head(h).squeeze(-1)
        return (pred, acts) if return_acts else pred


class TinyRes(nn.Module):
    def __init__(self, d: int, w: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.w1 = nn.Linear(d, w, bias=False)
        self.w2 = nn.Linear(w, d, bias=False)

    def forward(self, x):
        return x + self.w2(F.gelu(self.w1(x)))


class PairReplacedNet(nn.Module):
    def __init__(self, teacher, k: int, r1=None, r2=None, comp=None):
        super().__init__()
        self.stem = copy.deepcopy(teacher.stem)
        self.blocks = copy.deepcopy(teacher.blocks)
        self.head = copy.deepcopy(teacher.head)
        self.k = k
        self.r1 = r1
        self.r2 = r2
        self.comp = comp

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
        return self.head(h).squeeze(-1)


def datasets():
    X, y = load_diabetes(return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    idx = np.arange(len(y))

    train_idx, test_idx = train_test_split(
        idx, test_size=0.25, random_state=2234
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.2, random_state=6678
    )

    x_mean = X[train_idx].mean(axis=0, keepdims=True)
    x_std = X[train_idx].std(axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    y_mean = float(y[train_idx].mean())
    y_std = float(y[train_idx].std())
    if y_std < 1e-8:
        y_std = 1.0

    Xn = (X - x_mean) / x_std
    yn = (y - y_mean) / y_std

    return (
        torch.tensor(Xn[train_idx], dtype=torch.float32),
        torch.tensor(yn[train_idx], dtype=torch.float32),
        torch.tensor(Xn[val_idx], dtype=torch.float32),
        torch.tensor(yn[val_idx], dtype=torch.float32),
        torch.tensor(Xn[test_idx], dtype=torch.float32),
        torch.tensor(yn[test_idx], dtype=torch.float32),
        {
            "n_total": int(len(y)),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "input_dim": int(X.shape[1]),
            "feature_normalization": "train_mean_std_only",
            "target_normalization": "train_mean_std_only",
            "split_random_states": {"test": 2234, "val": 6678},
        },
    )


def train_model(seed: int, Xt, yt, input_dim: int, epochs: int = 60):
    model = Net(seed, input_dim=input_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 999)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), 64):
            ix = perm[i : i + 64]
            opt.zero_grad()
            loss = F.mse_loss(model(Xt[ix]), yt[ix])
            loss.backward()
            opt.step()
    return model


def mse(model, X, y):
    with torch.no_grad():
        return float(F.mse_loss(model(X), y))


def r2(model, X, y):
    with torch.no_grad():
        pred = model(X)
        sse = float(((pred - y) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum()) + 1e-12
        return 1.0 - sse / sst


def acts(model, X):
    with torch.no_grad():
        return model(X, return_acts=True)[1]


def fit_map(module, Xin, Yout, updates: int, seed: int):
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


def run(
    seed: int,
    *,
    nmse_threshold: float,
    r2_tolerance: float,
    grid: tuple[int, ...],
    teacher_epochs: int,
    map_updates: int,
):
    Xt, yt, Xv, yv, Xte, yte, data_meta = datasets()
    teacher = train_model(
        seed, Xt, yt, input_dim=data_meta["input_dim"], epochs=teacher_epochs
    )
    train_acts = acts(teacher, Xt)
    val_acts = acts(teacher, Xv)

    k = 0
    target = val_acts[k + 2]
    denom = float(((target - target.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    teacher_val_r2 = r2(teacher, Xv, yv)
    teacher_val_mse = mse(teacher, Xv, yv)

    rows = []
    selected_sep = None
    selected_comp = None

    for h in grid:
        # Each TinyRes has 2*d*w learned weights. With d=64:
        # separate: 2 * (128*h) = 256*h
        # composed at width 2h: 128*(2h) = 256*h
        budget = 256 * h
        r1 = fit_map(
            TinyRes(64, h, seed + 410000 + h),
            train_acts[k],
            train_acts[k + 1],
            map_updates,
            seed + 420000 + h,
        )
        r2m = fit_map(
            TinyRes(64, h, seed + 430000 + h),
            train_acts[k + 1],
            train_acts[k + 2],
            map_updates,
            seed + 440000 + h,
        )
        comp = fit_map(
            TinyRes(64, 2 * h, seed + 450000 + h),
            train_acts[k],
            train_acts[k + 2],
            map_updates,
            seed + 460000 + h,
        )

        sepnet = PairReplacedNet(teacher, k, r1=r1, r2=r2m)
        compnet = PairReplacedNet(teacher, k, comp=comp)
        with torch.no_grad():
            sep_nmse = float(F.mse_loss(r2m(r1(val_acts[k])), target)) / denom
            comp_nmse = float(F.mse_loss(comp(val_acts[k]), target)) / denom
        sep_val_r2 = r2(sepnet, Xv, yv)
        comp_val_r2 = r2(compnet, Xv, yv)

        row = {
            "h": h,
            "budget": budget,
            "sep_nmse": sep_nmse,
            "comp_nmse": comp_nmse,
            "sep_val_r2": sep_val_r2,
            "comp_val_r2": comp_val_r2,
            "sep_pass": bool(
                sep_nmse <= nmse_threshold
                and sep_val_r2 >= teacher_val_r2 - r2_tolerance
            ),
            "comp_pass": bool(
                comp_nmse <= nmse_threshold
                and comp_val_r2 >= teacher_val_r2 - r2_tolerance
            ),
        }

        # Test remains untouched until validation has selected an endpoint.
        if row["sep_pass"] and selected_sep is None:
            row["sep_test_r2"] = r2(sepnet, Xte, yte)
            row["sep_test_mse"] = mse(sepnet, Xte, yte)
            selected_sep = dict(row)
        if row["comp_pass"] and selected_comp is None:
            row["comp_test_r2"] = r2(compnet, Xte, yte)
            row["comp_test_mse"] = mse(compnet, Xte, yte)
            selected_comp = dict(row)
        rows.append(row)

    result = {
        "seed": seed,
        "dataset": "sklearn.datasets.load_diabetes",
        "task": "tabular_regression",
        "data": data_meta,
        "teacher_epochs": teacher_epochs,
        "map_updates": map_updates,
        "span": [0, 1],
        "internal_width": 64,
        "budget_grid_h": list(grid),
        "learned_parameter_budget_formula": "256*h for both conditions",
        "selection_rule": {
            "span_nmse_lte": nmse_threshold,
            "val_r2_gte_teacher_minus": r2_tolerance,
            "test_used_for_selection": False,
        },
        "teacher_val_r2": teacher_val_r2,
        "teacher_val_mse": teacher_val_mse,
        "grid": rows,
        "selected_sep": selected_sep,
        "selected_comp": selected_comp,
    }

    # Teacher test metrics are recorded only after candidate endpoint selection.
    result["teacher_test_r2"] = r2(teacher, Xte, yte)
    result["teacher_test_mse"] = mse(teacher, Xte, yte)

    if selected_sep is not None and selected_comp is not None:
        result["log2_budget_ratio"] = math.log2(
            selected_comp["budget"] / selected_sep["budget"]
        )
        result["test_r2_diff_comp_minus_sep"] = (
            selected_comp["comp_test_r2"] - selected_sep["sep_test_r2"]
        )
    else:
        result["log2_budget_ratio"] = None
        result["test_r2_diff_comp_minus_sep"] = None

    return result


def parse_grid(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values or any(x <= 0 for x in values):
        raise ValueError("grid must contain positive integer widths")
    if tuple(sorted(set(values))) != values:
        raise ValueError("grid must be strictly increasing with no duplicates")
    return values


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nmse-threshold", type=float, default=0.08)
    parser.add_argument("--r2-tolerance", type=float, default=0.05)
    parser.add_argument("--grid", type=str, default="2,4,6,8,12,16,24")
    parser.add_argument("--teacher-epochs", type=int, default=60)
    parser.add_argument("--map-updates", type=int, default=600)
    args = parser.parse_args()

    result = run(
        args.seed,
        nmse_threshold=args.nmse_threshold,
        r2_tolerance=args.r2_tolerance,
        grid=parse_grid(args.grid),
        teacher_epochs=args.teacher_epochs,
        map_updates=args.map_updates,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "seed": args.seed,
                "teacher_val_r2": result["teacher_val_r2"],
                "sep_budget": None
                if result["selected_sep"] is None
                else result["selected_sep"]["budget"],
                "comp_budget": None
                if result["selected_comp"] is None
                else result["selected_comp"]["budget"],
                "log2_ratio": result["log2_budget_ratio"],
            }
        )
    )
