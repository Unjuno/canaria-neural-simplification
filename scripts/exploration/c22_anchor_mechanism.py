from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exploration.c10_boundary_signal_ablation import (
    Chain,
    FullSpanReplacedNet,
    TinyRes,
    accuracy,
    acts,
    compile_final_from_hierarchy,
    count_params,
    fit_map,
    nmse,
    set_all_trainable,
    split_data,
    train_teacher,
)
from scripts.exploration.c11_interface_sketches import adapt_sketch


def fixed_projection():
    rng = np.random.default_rng(20261210)
    a = rng.standard_normal((64, 64))
    q, r = np.linalg.qr(a)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    q = q * signs[None, :]
    return torch.tensor(q[:, :16].astype(np.float32))


def adapt_target(module, a0t, target, updates, seed):
    set_all_trainable(module, True)
    params = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=8e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(updates):
        ix = torch.randint(0, len(a0t), (128,), generator=gen)
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(module(a0t[ix]), target[ix])
        loss.backward()
        opt.step()
    set_all_trainable(module, False)
    return module


def build_base(seed, a0t, a1t, a2t, a3t, a4t):
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
    hierarchy = Chain([copy.deepcopy(c12), copy.deepcopy(c34)])
    set_all_trainable(hierarchy, False)
    return hierarchy, locals4, c12, c34


def run(seed):
    Xt, yt, Xv, yv = split_data()
    teacher = train_teacher(seed, Xt, yt, 60)
    at, av = acts(teacher, Xt), acts(teacher, Xv)
    a0t, a1t, a2t, a3t, a4t = at
    a0v, _a1v, _a2v, _a3v, a4v = av
    denom = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    base, locals4, c12, c34 = build_base(seed, a0t, a1t, a2t, a3t, a4t)
    with torch.no_grad():
        baseline_t = base(a0t).detach()

    p16 = fixed_projection()
    mean_anchor = baseline_t.mean(0, keepdim=True).expand_as(baseline_t).detach()
    perm_gen = torch.Generator().manual_seed(seed + 180000)
    shuffled_anchor = baseline_t[torch.randperm(len(baseline_t), generator=perm_gen)].detach()
    anchors = {
        "anchor_self": baseline_t,
        "anchor_input": a0t.detach(),
        "anchor_mean": mean_anchor,
        "anchor_shuffled": shuffled_anchor,
        "anchor_zero": torch.zeros_like(baseline_t),
    }

    top_seed, init_seed, final_seed = seed + 150000, seed + 160000, seed + 170000
    conditions = {}
    for name in ["frozen", "sketch_only_16", "anchor_self", "anchor_input", "anchor_mean", "anchor_shuffled", "anchor_zero", "full_64"]:
        hierarchy = copy.deepcopy(base)
        if name == "sketch_only_16":
            adapt_sketch(hierarchy, a0t, a4t, p16, 600, top_seed)
        elif name in anchors:
            anchor = anchors[name]
            with torch.no_grad():
                target = (anchor + ((a4t - anchor) @ p16) @ p16.T).detach()
            adapt_target(hierarchy, a0t, target, 600, top_seed)
        elif name == "full_64":
            adapt_target(hierarchy, a0t, a4t, 600, top_seed)
        set_all_trainable(hierarchy, False)
        final, metrics = compile_final_from_hierarchy(hierarchy, a0t, a0v, a4v, denom, init_seed, final_seed)
        metrics["hierarchy_replacement_val_acc"] = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(hierarchy)), Xv, yv)
        metrics["final_replacement_val_acc"] = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(final)), Xv, yv)
        conditions[name] = metrics

    direct = fit_map(TinyRes(64, 32, init_seed), a0t, a4t, 600, final_seed)
    with torch.no_grad():
        direct_v = direct(a0v)
    direct_nmse = nmse(direct_v, a4v, denom)
    direct_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(direct)), Xv, yv)
    full_nmse = conditions["full_64"]["final_nmse_vs_original"]
    frozen_nmse = conditions["frozen"]["final_nmse_vs_original"]
    for rec in conditions.values():
        rec["ratio_final_nmse_over_direct"] = rec["final_nmse_vs_original"] / direct_nmse
        rec["ratio_final_nmse_over_full64"] = rec["final_nmse_vs_original"] / full_nmse
        rec["difference_final_nmse_vs_frozen"] = rec["final_nmse_vs_original"] - frozen_nmse

    return {
        "seed": seed,
        "status": "EXPLORATORY_OUTCOME",
        "test_evaluated": False,
        "teacher_val_acc": accuracy(teacher, Xv, yv),
        "basis": {"rng_seed": 20261210, "dimension": 16},
        "shuffle_rule": {"generator_seed": seed + 180000},
        "budget": {
            "local_total_params": sum(count_params(m) for m in locals4),
            "level1_total_params": count_params(c12) + count_params(c34),
            "final_params": count_params(direct),
            "exact_4096_each_level": sum(count_params(m) for m in locals4) == count_params(c12) + count_params(c34) == count_params(direct) == 4096,
        },
        "conditions": conditions,
        "direct_original_single": {"final_nmse_vs_original": direct_nmse, "replacement_val_acc": direct_acc},
        "ordering_final_nmse_best_to_worst": sorted(conditions, key=lambda k: conditions[k]["final_nmse_vs_original"]),
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
