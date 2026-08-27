from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[0]

# These primitives are copied from the locked C6 family implementation to keep C10 standalone.
torch.set_num_threads(1)


class ResBlock(torch.nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.ln = torch.nn.LayerNorm(d)
        self.fc1 = torch.nn.Linear(d, 128)
        self.fc2 = torch.nn.Linear(128, d)

    def forward(self, x):
        z = self.ln(x)
        z = F.gelu(self.fc1(z))
        z = self.fc2(z)
        return x + z


class Net(torch.nn.Module):
    def __init__(self, seed, d=64, depth=4):
        super().__init__()
        torch.manual_seed(seed)
        self.stem = torch.nn.Linear(64, d)
        self.blocks = torch.nn.ModuleList([ResBlock(d) for _ in range(depth)])
        self.head = torch.nn.Linear(d, 10)

    def forward(self, x, return_acts=False):
        h = F.gelu(self.stem(x))
        acts = [h]
        for b in self.blocks:
            h = b(h)
            acts.append(h)
        logits = self.head(h)
        return (logits, acts) if return_acts else logits


class TinyRes(torch.nn.Module):
    def __init__(self, d, w, seed):
        super().__init__()
        torch.manual_seed(seed)
        self.w1 = torch.nn.Linear(d, w, bias=False)
        self.w2 = torch.nn.Linear(w, d, bias=False)

    def forward(self, x):
        return x + self.w2(F.gelu(self.w1(x)))


class Chain(torch.nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules_seq = torch.nn.ModuleList(modules)

    def forward(self, x):
        for m in self.modules_seq:
            x = m(x)
        return x


class FullSpanReplacedNet(torch.nn.Module):
    def __init__(self, teacher, replacement):
        super().__init__()
        self.stem = copy.deepcopy(teacher.stem)
        self.head = copy.deepcopy(teacher.head)
        self.replacement = replacement
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
        train_idx, test_size=0.2, random_state=5678, stratify=y[train_idx]
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
            ix = perm[i:i+64]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[ix]), yt[ix])
            loss.backward()
            opt.step()
    return model


def acts(model, X):
    with torch.no_grad():
        return model(X, return_acts=True)[1]


def accuracy(model, X, y):
    with torch.no_grad():
        return float((model(X).argmax(-1) == y).float().mean())


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


def set_all_trainable(module, required):
    for p in module.parameters():
        p.requires_grad_(required)
    return module


def nmse(pred, target, denom=None):
    if denom is None:
        denom = float(((target - target.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    return float(F.mse_loss(pred, target)) / denom


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def adapt_top_boundary(module, mode, a0t, a4t, yt, teacher_head, teacher_logits_t, updates, seed):
    set_all_trainable(module, True)
    params = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=8e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    n = len(a0t)
    for _ in range(updates):
        ix = torch.randint(0, n, (128,), generator=gen)
        opt.zero_grad()
        h = module(a0t[ix])
        if mode == "hidden_mse":
            loss = F.mse_loss(h, a4t[ix])
        elif mode == "logit_distill":
            pred_logits = teacher_head(h)
            loss = F.mse_loss(pred_logits, teacher_logits_t[ix])
        elif mode == "label_ce":
            pred_logits = teacher_head(h)
            loss = F.cross_entropy(pred_logits, yt[ix])
        else:
            raise ValueError(mode)
        loss.backward()
        opt.step()
    set_all_trainable(module, False)
    return module


def compile_final_from_hierarchy(hierarchy, a0t, a0v, a4v, denom_original, init_seed, fit_seed):
    with torch.no_grad():
        target_t = hierarchy(a0t).detach()
        target_v = hierarchy(a0v).detach()
    final = TinyRes(64, 32, init_seed)
    final = fit_map(final, a0t, target_t, 600, fit_seed)
    with torch.no_grad():
        out_v = final(a0v)
    denom_hierarchy = float(((target_v - target_v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    return final, {
        "hierarchy_nmse_vs_original": nmse(target_v, a4v, denom_original),
        "final_nmse_vs_hierarchy": nmse(out_v, target_v, denom_hierarchy),
        "final_nmse_vs_original": nmse(out_v, a4v, denom_original),
    }


def run(seed):
    Xt, yt, Xv, yv = split_data()
    teacher = train_teacher(seed, Xt, yt, 60)
    at = acts(teacher, Xt)
    av = acts(teacher, Xv)
    a0t, a1t, a2t, a3t, a4t = at
    a0v, _a1v, a2v, _a3v, a4v = av
    denom_full = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    locals4 = [
        fit_map(TinyRes(64, 8, seed + 101001), a0t, a1t, 600, seed + 102001),
        fit_map(TinyRes(64, 8, seed + 101002), a1t, a2t, 600, seed + 102002),
        fit_map(TinyRes(64, 8, seed + 101003), a2t, a3t, 600, seed + 102003),
        fit_map(TinyRes(64, 8, seed + 101004), a3t, a4t, 600, seed + 102004),
    ]

    pair12 = Chain([copy.deepcopy(locals4[0]), copy.deepcopy(locals4[1])])
    set_all_trainable(pair12, True)
    fit_map(pair12, a0t, a2t, 600, seed + 110001)
    set_all_trainable(pair12, False)

    pair34 = Chain([copy.deepcopy(locals4[2]), copy.deepcopy(locals4[3])])
    set_all_trainable(pair34, True)
    fit_map(pair34, a2t, a4t, 600, seed + 110002)
    set_all_trainable(pair34, False)

    with torch.no_grad():
        pair12_t = pair12(a0t).detach()
        pair34_t = pair34(a2t).detach()
    c12 = fit_map(TinyRes(64, 16, seed + 120001), a0t, pair12_t, 600, seed + 121001)
    c34 = fit_map(TinyRes(64, 16, seed + 120002), a2t, pair34_t, 600, seed + 121002)

    base_hierarchy = Chain([copy.deepcopy(c12), copy.deepcopy(c34)])
    set_all_trainable(base_hierarchy, False)

    with torch.no_grad():
        teacher_logits_t = teacher.head(a4t).detach()

    head = copy.deepcopy(teacher.head)
    for p in head.parameters():
        p.requires_grad_(False)

    top_seed = seed + 150000
    final_init_seed = seed + 160000
    final_fit_seed = seed + 170000

    conditions = {}
    modes = ["frozen", "hidden_mse", "logit_distill", "label_ce"]
    for mode in modes:
        hierarchy = copy.deepcopy(base_hierarchy)
        if mode != "frozen":
            adapt_top_boundary(
                hierarchy,
                mode,
                a0t,
                a4t,
                yt,
                head,
                teacher_logits_t,
                600,
                top_seed,
            )
        set_all_trainable(hierarchy, False)
        pre_val_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(hierarchy)), Xv, yv)
        final, metrics = compile_final_from_hierarchy(
            hierarchy, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
        )
        metrics["hierarchy_replacement_val_acc"] = pre_val_acc
        metrics["final_replacement_val_acc"] = accuracy(
            FullSpanReplacedNet(teacher, copy.deepcopy(final)), Xv, yv
        )
        conditions[mode] = metrics

    direct = TinyRes(64, 32, final_init_seed)
    direct = fit_map(direct, a0t, a4t, 600, final_fit_seed)
    with torch.no_grad():
        direct_v = direct(a0v)
    direct_nmse = nmse(direct_v, a4v, denom_full)
    direct_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(direct)), Xv, yv)

    for rec in conditions.values():
        rec["ratio_final_nmse_over_direct"] = rec["final_nmse_vs_original"] / direct_nmse

    return {
        "seed": seed,
        "status": "EXPLORATORY_OUTCOME",
        "test_evaluated": False,
        "teacher_val_acc": accuracy(teacher, Xv, yv),
        "budget": {
            "local_total_params": sum(count_params(m) for m in locals4),
            "level1_total_params": count_params(c12) + count_params(c34),
            "final_params": count_params(direct),
            "exact_4096_each_level": (
                sum(count_params(m) for m in locals4)
                == count_params(c12) + count_params(c34)
                == count_params(direct)
                == 4096
            ),
        },
        "conditions": conditions,
        "direct_original_single": {
            "final_nmse_vs_original": direct_nmse,
            "replacement_val_acc": direct_acc,
        },
        "ordering_final_nmse_best_to_worst": sorted(
            conditions, key=lambda k: conditions[k]["final_nmse_vs_original"]
        ),
        "ordering_final_val_acc_best_to_worst": sorted(
            conditions, key=lambda k: -conditions[k]["final_replacement_val_acc"]
        ),
    }


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
