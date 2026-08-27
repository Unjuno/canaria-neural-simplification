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
        for b in self.blocks:
            h = b(h)
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


class Chain(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules_seq = nn.ModuleList(modules)

    def forward(self, x):
        for m in self.modules_seq:
            x = m(x)
        return x


class FullSpanReplacedNet(nn.Module):
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


def final_from_recursive_teacher(seed, teacher_module, a0t, a0v, a4v, denom_original, init_seed, fit_seed):
    with torch.no_grad():
        target_t = teacher_module(a0t).detach()
        target_v = teacher_module(a0v).detach()
    final = TinyRes(64, 32, init_seed)
    final = fit_map(final, a0t, target_t, 600, fit_seed)
    with torch.no_grad():
        out_v = final(a0v)
    denom_recursive = float(((target_v - target_v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    return final, {
        "recursive_teacher_nmse_vs_original": nmse(target_v, a4v, denom_original),
        "final_nmse_vs_recursive_teacher": nmse(out_v, target_v, denom_recursive),
        "final_nmse_vs_original": nmse(out_v, a4v, denom_original),
    }


def run(seed):
    Xt, yt, Xv, yv = split_data()
    teacher = train_teacher(seed, Xt, yt)
    at = acts(teacher, Xt)
    av = acts(teacher, Xv)
    a0t, a1t, a2t, a3t, a4t = at
    a0v, a1v, a2v, a3v, a4v = av
    denom_full = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    locals4 = [
        fit_map(TinyRes(64, 8, seed + 101001), a0t, a1t, 600, seed + 102001),
        fit_map(TinyRes(64, 8, seed + 101002), a1t, a2t, 600, seed + 102002),
        fit_map(TinyRes(64, 8, seed + 101003), a2t, a3t, 600, seed + 102003),
        fit_map(TinyRes(64, 8, seed + 101004), a3t, a4t, 600, seed + 102004),
    ]

    # Level 1: jointly adapt each local pair against its original two-block span.
    pair12 = Chain([copy.deepcopy(locals4[0]), copy.deepcopy(locals4[1])])
    set_all_trainable(pair12, True)
    fit_map(pair12, a0t, a2t, 600, seed + 110001)
    set_all_trainable(pair12, False)

    pair34 = Chain([copy.deepcopy(locals4[2]), copy.deepcopy(locals4[3])])
    set_all_trainable(pair34, True)
    fit_map(pair34, a2t, a4t, 600, seed + 110002)
    set_all_trainable(pair34, False)

    # Recursive Level-1 compilation: only pair-cluster outputs are fit targets.
    with torch.no_grad():
        pair12_t = pair12(a0t).detach()
        pair34_t = pair34(a2t).detach()
    c12 = fit_map(TinyRes(64, 16, seed + 120001), a0t, pair12_t, 600, seed + 121001)
    c34 = fit_map(TinyRes(64, 16, seed + 120002), a2t, pair34_t, 600, seed + 121002)

    with torch.no_grad():
        c12_v = c12(a0v)
        c34_v_on_original_input = c34(a2v)
    denom_a2 = float(((a2v - a2v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    denom_a4 = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    level1 = {
        "C12_nmse_vs_original_pair": nmse(c12_v, a2v, denom_a2),
        "C34_nmse_vs_original_pair": nmse(c34_v_on_original_input, a4v, denom_a4),
        "pair12_params": count_params(c12),
        "pair34_params": count_params(c34),
    }

    base_hierarchy = Chain([copy.deepcopy(c12), copy.deepcopy(c34)])
    set_all_trainable(base_hierarchy, False)

    final_init_seed = seed + 130000
    final_fit_seed = seed + 140000
    conditions = {}

    # A: strict depth-2 recursion: no return to original full-span target at Level 2.
    hierarchy_frozen = copy.deepcopy(base_hierarchy)
    final_hf, m_hf = final_from_recursive_teacher(
        seed, hierarchy_frozen, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
    )
    m_hf["replacement_val_acc"] = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(final_hf)), Xv, yv)
    conditions["hierarchical_frozen"] = m_hf

    # B: adapt recursively generated C12+C34 jointly to original full span, then freeze and recurse.
    hierarchy_joint = copy.deepcopy(base_hierarchy)
    set_all_trainable(hierarchy_joint, True)
    fit_map(hierarchy_joint, a0t, a4t, 600, seed + 150000)
    set_all_trainable(hierarchy_joint, False)
    final_hj, m_hj = final_from_recursive_teacher(
        seed, hierarchy_joint, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
    )
    m_hj["replacement_val_acc"] = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(final_hj)), Xv, yv)
    conditions["hierarchical_joint"] = m_hj

    # C: one-level recursive baseline from all four separately fitted local candidates.
    single_level_cluster = Chain([copy.deepcopy(m) for m in locals4])
    set_all_trainable(single_level_cluster, True)
    fit_map(single_level_cluster, a0t, a4t, 600, seed + 160000)
    set_all_trainable(single_level_cluster, False)
    final_sl, m_sl = final_from_recursive_teacher(
        seed, single_level_cluster, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
    )
    m_sl["replacement_val_acc"] = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(final_sl)), Xv, yv)
    conditions["single_level_recursive"] = m_sl

    # D: matched direct-original control.
    direct = TinyRes(64, 32, final_init_seed)
    direct = fit_map(direct, a0t, a4t, 600, final_fit_seed)
    with torch.no_grad():
        direct_v = direct(a0v)
    direct_nmse = nmse(direct_v, a4v, denom_full)
    direct_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(direct)), Xv, yv)

    for rec in conditions.values():
        rec["ratio_final_nmse_over_direct"] = rec["final_nmse_vs_original"] / direct_nmse
    conditions["hierarchical_frozen"]["ratio_over_single_level_recursive"] = (
        conditions["hierarchical_frozen"]["final_nmse_vs_original"]
        / conditions["single_level_recursive"]["final_nmse_vs_original"]
    )
    conditions["hierarchical_joint"]["ratio_over_single_level_recursive"] = (
        conditions["hierarchical_joint"]["final_nmse_vs_original"]
        / conditions["single_level_recursive"]["final_nmse_vs_original"]
    )

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
                sum(count_params(m) for m in locals4) == count_params(c12) + count_params(c34) == count_params(direct) == 4096
            ),
        },
        "level1": level1,
        "conditions": conditions,
        "direct_original_single": {
            "final_nmse_vs_original": direct_nmse,
            "replacement_val_acc": direct_acc,
        },
        "ordering_best_to_worst": sorted(
            conditions,
            key=lambda k: conditions[k]["final_nmse_vs_original"],
        ) + ["direct_original_single"],
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
