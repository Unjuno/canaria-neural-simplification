from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


def fixed_basis():
    rng = np.random.default_rng(20260831)
    a = rng.standard_normal((64, 64))
    q, r = np.linalg.qr(a)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    q = q * signs[None, :]
    return torch.tensor(q.astype(np.float32))


def adapt_sketch(module, a0t, a4t, projection, updates, seed):
    set_all_trainable(module, True)
    params = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=8e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    n = len(a0t)
    target_proj = a4t @ projection
    for _ in range(updates):
        ix = torch.randint(0, n, (128,), generator=gen)
        opt.zero_grad()
        pred_proj = module(a0t[ix]) @ projection
        loss = F.mse_loss(pred_proj, target_proj[ix])
        loss.backward()
        opt.step()
    set_all_trainable(module, False)
    return module


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

    basis = fixed_basis()
    top_seed = seed + 150000
    final_init_seed = seed + 160000
    final_fit_seed = seed + 170000

    conditions = {}
    dims = [4, 8, 16, 32, 64]
    all_conditions = ["frozen"] + ["full_64" if k == 64 else f"sketch_{k}" for k in dims]
    for name in all_conditions:
        hierarchy = copy.deepcopy(base_hierarchy)
        if name != "frozen":
            k = 64 if name == "full_64" else int(name.split("_")[1])
            adapt_sketch(hierarchy, a0t, a4t, basis[:, :k], 600, top_seed)
        set_all_trainable(hierarchy, False)
        final, metrics = compile_final_from_hierarchy(
            hierarchy, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
        )
        metrics["hierarchy_replacement_val_acc"] = accuracy(
            FullSpanReplacedNet(teacher, copy.deepcopy(hierarchy)), Xv, yv
        )
        metrics["final_replacement_val_acc"] = accuracy(
            FullSpanReplacedNet(teacher, copy.deepcopy(final)), Xv, yv
        )
        conditions[name] = metrics

    direct = TinyRes(64, 32, final_init_seed)
    direct = fit_map(direct, a0t, a4t, 600, final_fit_seed)
    with torch.no_grad():
        direct_v = direct(a0v)
    direct_nmse = nmse(direct_v, a4v, denom_full)
    direct_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(direct)), Xv, yv)

    full_nmse = conditions["full_64"]["final_nmse_vs_original"]
    for rec in conditions.values():
        rec["ratio_final_nmse_over_direct"] = rec["final_nmse_vs_original"] / direct_nmse
        rec["ratio_final_nmse_over_full64"] = rec["final_nmse_vs_original"] / full_nmse

    return {
        "seed": seed,
        "status": "EXPLORATORY_OUTCOME",
        "test_evaluated": False,
        "teacher_val_acc": accuracy(teacher, Xv, yv),
        "basis": {"rng_seed": 20260831, "nested_dimensions": dims},
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
