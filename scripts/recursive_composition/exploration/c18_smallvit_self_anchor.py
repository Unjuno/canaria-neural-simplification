from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recursive_composition.exploration.c4_smallvit_recursive import (
    Cluster,
    SmallViT,
    TinyTokenRes,
    accuracy,
    count_params,
    data_split,
    fit_map,
    set_seed,
    train_teacher,
)


torch.set_num_threads(1)


class ReplacedFullViT(nn.Module):
    def __init__(self, teacher: SmallViT, replacement: nn.Module):
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


@torch.no_grad()
def collect_full_span(model: SmallViT, ds):
    model.eval()
    outs = [[] for _ in range(5)]
    for x, _ in torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False):
        h = model.embed(x)
        outs[0].append(h.cpu())
        for i, block in enumerate(model.blocks):
            h = block(h)
            outs[i + 1].append(h.cpu())
    return [torch.cat(v) for v in outs]


def set_all_trainable(module: nn.Module, required: bool):
    for p in module.parameters():
        p.requires_grad_(required)
    return module


def nmse(pred, target, denom=None):
    if denom is None:
        denom = float(((target - target.mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12
    return float(F.mse_loss(pred, target)) / denom


def fixed_basis():
    rng = np.random.default_rng(20261020)
    a = rng.standard_normal((32, 32))
    q, r = np.linalg.qr(a)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    q = q * signs[None, :]
    return torch.tensor(q.astype(np.float32))


def adapt_sketch(module, xin, target, projection, updates, seed):
    set_all_trainable(module, True)
    params = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=8e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    n = len(xin)
    target_proj = target @ projection
    for _ in range(updates):
        ix = torch.randint(0, n, (64,), generator=gen)
        opt.zero_grad()
        pred = module(xin[ix]) @ projection
        loss = F.mse_loss(pred, target_proj[ix])
        loss.backward()
        opt.step()
    set_all_trainable(module, False)
    return module


def adapt_full_target(module, xin, target, updates, seed):
    set_all_trainable(module, True)
    params = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=8e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    n = len(xin)
    for _ in range(updates):
        ix = torch.randint(0, n, (64,), generator=gen)
        opt.zero_grad()
        loss = F.mse_loss(module(xin[ix]), target[ix])
        loss.backward()
        opt.step()
    set_all_trainable(module, False)
    return module


def compile_from_frozen(hierarchy, a0fit, a0v, a4v, teacher_denom, init_seed, fit_seed):
    with torch.no_grad():
        target_fit = hierarchy(a0fit).detach()
        target_val = hierarchy(a0v).detach()
    final = TinyTokenRes(32, 64, init_seed)
    final = fit_map(final, a0fit, target_fit, 600, fit_seed)
    with torch.no_grad():
        out_v = final(a0v)
    hierarchy_denom = float(
        ((target_val - target_val.mean(dim=(0, 1), keepdim=True)) ** 2).mean()
    ) + 1e-12
    return final, {
        "hierarchy_nmse_vs_original": nmse(target_val, a4v, teacher_denom),
        "final_nmse_vs_hierarchy": nmse(out_v, target_val, hierarchy_denom),
        "final_nmse_vs_original": nmse(out_v, a4v, teacher_denom),
    }


def run(seed: int):
    set_seed(seed)
    tr, va = data_split()
    teacher = train_teacher(SmallViT(depth=4), tr, seed + 50000, 45)
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

    at = collect_full_span(teacher, tr)
    av = collect_full_span(teacher, va)
    assert len(at) == len(av) == 5
    fit = [a[:512] for a in at]
    teacher_denom = float(((av[4] - av[4].mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12

    local = []
    for i in range(4):
        local.append(
            fit_map(
                TinyTokenRes(32, 16, seed + 101000 + i),
                fit[i],
                fit[i + 1],
                600,
                seed + 102000 + i,
            )
        )

    level1 = []
    for g, start in enumerate((0, 2)):
        pair = Cluster([copy.deepcopy(local[start]), copy.deepcopy(local[start + 1])])
        set_all_trainable(pair, True)
        fit_map(pair, fit[start], fit[start + 2], 600, seed + 110000 + g)
        set_all_trainable(pair, False)
        with torch.no_grad():
            pair_target = pair(fit[start]).detach()
        unit = TinyTokenRes(32, 32, seed + 120000 + g)
        unit = fit_map(unit, fit[start], pair_target, 600, seed + 121000 + g)
        level1.append(unit)

    base = Cluster([copy.deepcopy(level1[0]), copy.deepcopy(level1[1])])
    set_all_trainable(base, False)
    with torch.no_grad():
        baseline_fit = base(fit[0]).detach()

    assert sum(count_params(x) for x in local) == 4096
    assert sum(count_params(x) for x in level1) == 4096

    basis = fixed_basis()
    p8 = basis[:, :8]
    p16 = basis[:, :16]
    top_seed = seed + 150000
    final_init_seed = seed + 160000
    final_fit_seed = seed + 170000

    conditions = {}
    for name in ("frozen", "sketch_only_16", "anchored_8", "anchored_16", "full_32"):
        hierarchy = copy.deepcopy(base)
        if name == "sketch_only_16":
            adapt_sketch(hierarchy, fit[0], fit[4], p16, 600, top_seed)
        elif name in ("anchored_8", "anchored_16"):
            p = p8 if name == "anchored_8" else p16
            with torch.no_grad():
                correction = ((fit[4] - baseline_fit) @ p) @ p.T
                hybrid_target = (baseline_fit + correction).detach()
            adapt_full_target(hierarchy, fit[0], hybrid_target, 600, top_seed)
        elif name == "full_32":
            adapt_full_target(hierarchy, fit[0], fit[4], 600, top_seed)

        set_all_trainable(hierarchy, False)
        final, metrics = compile_from_frozen(
            hierarchy,
            fit[0],
            av[0],
            av[4],
            teacher_denom,
            final_init_seed,
            final_fit_seed,
        )
        metrics["hierarchy_replacement_val_acc"] = accuracy(
            ReplacedFullViT(teacher, copy.deepcopy(hierarchy)), va
        )
        metrics["final_replacement_val_acc"] = accuracy(
            ReplacedFullViT(teacher, copy.deepcopy(final)), va
        )
        conditions[name] = metrics

    direct = TinyTokenRes(32, 64, final_init_seed)
    direct = fit_map(direct, fit[0], fit[4], 600, final_fit_seed)
    assert count_params(direct) == 4096
    with torch.no_grad():
        direct_v = direct(av[0])
    direct_nmse = nmse(direct_v, av[4], teacher_denom)
    direct_acc = accuracy(ReplacedFullViT(teacher, copy.deepcopy(direct)), va)

    full_nmse = conditions["full_32"]["final_nmse_vs_original"]
    frozen_nmse = conditions["frozen"]["final_nmse_vs_original"]
    for rec in conditions.values():
        rec["ratio_final_over_direct"] = rec["final_nmse_vs_original"] / direct_nmse
        rec["ratio_final_over_full32"] = rec["final_nmse_vs_original"] / full_nmse
        rec["difference_final_vs_frozen"] = rec["final_nmse_vs_original"] - frozen_nmse

    result.update(
        {
            "budget": {
                "level0_total_params": sum(count_params(x) for x in local),
                "level1_total_params": sum(count_params(x) for x in level1),
                "final_params": count_params(direct),
                "exact_4096_each_level": (
                    sum(count_params(x) for x in local)
                    == sum(count_params(x) for x in level1)
                    == count_params(direct)
                    == 4096
                ),
            },
            "basis": {"rng_seed": 20261020, "nested_dimensions": [8, 16]},
            "conditions": conditions,
            "direct_original_single": {
                "final_nmse_vs_original": direct_nmse,
                "replacement_val_acc": direct_acc,
            },
            "ordering_final_nmse_best_to_worst": sorted(
                conditions, key=lambda n: conditions[n]["final_nmse_vs_original"]
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
