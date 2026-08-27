from __future__ import annotations

import argparse
import copy
import json
import math
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


class TinyAttnRes(nn.Module):
    def __init__(self, d: int, width: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.width = width
        self.q = nn.Linear(d, width, bias=False)
        self.k = nn.Linear(d, width, bias=False)
        self.v = nn.Linear(d, width, bias=False)
        self.o = nn.Linear(width, d, bias=False)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.width)
        attn = torch.softmax(score, dim=-1)
        return x + self.o(torch.matmul(attn, v))


class Cluster(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.mods = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.mods:
            x = module(x)
        return x


class WholeReplacedViT(nn.Module):
    def __init__(self, teacher, replacement):
        super().__init__()
        self.patch = copy.deepcopy(teacher.patch)
        self.cls = nn.Parameter(teacher.cls.detach().clone(), requires_grad=False)
        self.pos = nn.Parameter(teacher.pos.detach().clone(), requires_grad=False)
        self.replacement = replacement
        self.norm = copy.deepcopy(teacher.norm)
        self.head = copy.deepcopy(teacher.head)
        for module in (self.patch, self.norm, self.head):
            for p in module.parameters():
                p.requires_grad_(False)

    def embed(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        c = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([c, x], 1) + self.pos

    def forward(self, x):
        h = self.replacement(self.embed(x))
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
def collect_acts(model, ds):
    model.eval()
    accum = [[] for _ in range(5)]
    for x, _ in torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False):
        h = model.embed(x)
        accum[0].append(h.cpu())
        for i, block in enumerate(model.blocks):
            h = block(h)
            accum[i + 1].append(h.cpu())
    return [torch.cat(v) for v in accum]


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


def set_all_trainable(module, value):
    for p in module.parameters():
        p.requires_grad_(value)
    return module


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def nmse(pred, target, denom=None):
    if denom is None:
        denom = float(((target - target.mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12
    return float(F.mse_loss(pred, target)) / denom


def final_fit(seed, target_t, target_v, a0t, a0v, a4v, original_denom, teacher, va):
    single = TinyAttnRes(32, 64, seed + 160001)
    single = fit_map(single, a0t, target_t, 600, seed + 170001)
    with torch.no_grad():
        out = single(a0v)
    immediate_denom = float(((target_v - target_v.mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12
    return single, {
        "nmse_vs_immediate_canaria_teacher": nmse(out, target_v, immediate_denom),
        "nmse_vs_original_teacher": nmse(out, a4v, original_denom),
        "replacement_val_acc": accuracy(WholeReplacedViT(teacher, copy.deepcopy(single)), va),
    }


def run(seed: int):
    set_seed(seed)
    tr, va = data_split()
    teacher = train_teacher(SmallViT(), tr, seed + 50000, 45)
    teacher_val = accuracy(teacher, va)
    result = {"seed": seed, "status": "EXPLORATORY_OUTCOME", "teacher_val_acc": teacher_val, "eligible": bool(teacher_val >= 0.95), "test_evaluated": False}
    if not result["eligible"]:
        return result

    at = collect_acts(teacher, tr)
    av = collect_acts(teacher, va)
    a0t, a1t, a2t, a3t, a4t = [x[:512] for x in at]
    a0v, a1v, a2v, a3v, a4v = av
    original_denom = float(((a4v - a4v.mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12

    locals4 = [
        fit_map(TinyAttnRes(32,16,seed+101001), a0t,a1t,600,seed+102001),
        fit_map(TinyAttnRes(32,16,seed+101002), a1t,a2t,600,seed+102002),
        fit_map(TinyAttnRes(32,16,seed+101003), a2t,a3t,600,seed+102003),
        fit_map(TinyAttnRes(32,16,seed+101004), a3t,a4t,600,seed+102004),
    ]
    pair12 = Cluster([copy.deepcopy(locals4[0]), copy.deepcopy(locals4[1])])
    pair34 = Cluster([copy.deepcopy(locals4[2]), copy.deepcopy(locals4[3])])
    set_all_trainable(pair12, True); set_all_trainable(pair34, True)
    pair12 = fit_map(pair12, a0t, a2t, 600, seed+120001)
    pair34 = fit_map(pair34, a2t, a4t, 600, seed+120002)
    set_all_trainable(pair12, False); set_all_trainable(pair34, False)
    with torch.no_grad():
        p12t,p12v = pair12(a0t).detach(), pair12(a0v).detach()
        p34t = pair34(a2t).detach()
        lower_t,lower_v = pair34(p12t).detach(), pair34(p12v).detach()

    c12 = fit_map(TinyAttnRes(32,32,seed+130001), a0t,p12t,600,seed+140001)
    c34 = fit_map(TinyAttnRes(32,32,seed+130002), a2t,p34t,600,seed+140002)
    with torch.no_grad():
        pre_t,pre_v = c34(c12(a0t)).detach(), c34(c12(a0v)).detach()

    l2 = Cluster([copy.deepcopy(c12), copy.deepcopy(c34)])
    set_all_trainable(l2, True)
    l2 = fit_map(l2, a0t, lower_t, 600, seed+150001)
    set_all_trainable(l2, False)
    with torch.no_grad():
        post_t,post_v = l2(a0t).detach(), l2(a0v).detach()

    finals = {}
    _, finals["hierarchical_no_level2_adapt"] = final_fit(seed, pre_t,pre_v,a0t,a0v,a4v,original_denom,teacher,va)
    _, finals["hierarchical_level2_joint_adapt"] = final_fit(seed, post_t,post_v,a0t,a0v,a4v,original_denom,teacher,va)
    _, finals["flat_lower_ir_single"] = final_fit(seed, lower_t,lower_v,a0t,a0v,a4v,original_denom,teacher,va)
    _, finals["direct_original_single"] = final_fit(seed, a4t,a4v,a0t,a0v,a4v,original_denom,teacher,va)

    lower_denom = float(((lower_v-lower_v.mean(dim=(0,1),keepdim=True))**2).mean()) + 1e-12
    with torch.no_grad():
        post_v_now = l2(a0v)
    joint = finals["hierarchical_level2_joint_adapt"]
    no_adapt = finals["hierarchical_no_level2_adapt"]
    flat = finals["flat_lower_ir_single"]
    direct = finals["direct_original_single"]
    result.update({
        "fit_examples": 512,
        "budget": {"level0_total_params": sum(count_params(m) for m in locals4), "level1_total_params": count_params(c12)+count_params(c34), "level2_params": count_params(TinyAttnRes(32,64,0)), "exact_8192_match": sum(count_params(m) for m in locals4) == count_params(c12)+count_params(c34) == count_params(TinyAttnRes(32,64,0)) == 8192},
        "structural": {"lower_ir_nmse_vs_original": nmse(lower_v,a4v,original_denom), "post_level2_pair_nmse_vs_lower_ir": nmse(post_v_now,lower_v,lower_denom), "post_level2_pair_nmse_vs_original": nmse(post_v_now,a4v,original_denom)},
        "finals": finals,
        "derived": {"D_joint_minus_no_adapt": joint["nmse_vs_original_teacher"]-no_adapt["nmse_vs_original_teacher"], "R_joint_over_flat": joint["nmse_vs_original_teacher"]/flat["nmse_vs_original_teacher"], "R_joint_over_direct": joint["nmse_vs_original_teacher"]/direct["nmse_vs_original_teacher"], "val_acc_diff_joint_minus_flat": joint["replacement_val_acc"]-flat["replacement_val_acc"]},
        "final_order_best_to_worst": sorted(finals, key=lambda k: finals[k]["nmse_vs_original_teacher"]),
    })
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args(); rec=run(args.seed); args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(rec,indent=2),encoding="utf-8"); print(json.dumps(rec,indent=2))


if __name__ == "__main__":
    main()
