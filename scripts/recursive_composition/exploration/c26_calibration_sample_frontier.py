from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import (
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


torch.set_num_threads(1)
CAL_COUNTS = [8, 16, 32, 64, 128, 256, 512]


def projection16():
    rng = np.random.default_rng(20261410)
    a = rng.standard_normal((64, 64))
    q, r = np.linalg.qr(a)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    q = q * signs[None, :]
    return torch.tensor(q[:, :16].astype(np.float32))


def adapt_on_subset(module, xin, target, updates, seed):
    set_all_trainable(module, True)
    params = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=8e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    n = len(xin)
    for _ in range(updates):
        ix = torch.randint(0, n, (128,), generator=gen)
        opt.zero_grad()
        loss = F.mse_loss(module(xin[ix]), target[ix])
        loss.backward()
        opt.step()
    set_all_trainable(module, False)
    return module


def run(seed: int):
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

    base = Chain([copy.deepcopy(c12), copy.deepcopy(c34)])
    set_all_trainable(base, False)
    with torch.no_grad():
        baseline_t = base(a0t).detach()

    p = projection16()
    with torch.no_grad():
        hybrid_t = (baseline_t + ((a4t - baseline_t) @ p) @ p.T).detach()

    order_gen = torch.Generator().manual_seed(seed + 190000)
    order = torch.randperm(len(a0t), generator=order_gen)
    top_seed = seed + 150000
    final_init_seed = seed + 160000
    final_fit_seed = seed + 170000

    conditions = {}

    frozen = copy.deepcopy(base)
    frozen_final, frozen_m = compile_final_from_hierarchy(
        frozen, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
    )
    frozen_m["final_replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(frozen_final)), Xv, yv
    )
    conditions["frozen"] = frozen_m

    count_specs = [(str(k), k) for k in CAL_COUNTS] + [("all", len(a0t))]
    for label, k in count_specs:
        ix = order[:k]
        hierarchy = copy.deepcopy(base)
        adapt_on_subset(hierarchy, a0t[ix], hybrid_t[ix], 600, top_seed)
        final, metrics = compile_final_from_hierarchy(
            hierarchy, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
        )
        metrics["calibration_unique_samples"] = int(k)
        metrics["final_replacement_val_acc"] = accuracy(
            FullSpanReplacedNet(teacher, copy.deepcopy(final)), Xv, yv
        )
        conditions[f"self_{label}"] = metrics

    full = copy.deepcopy(base)
    adapt_on_subset(full, a0t[order], a4t[order].detach(), 600, top_seed)
    full_final, full_m = compile_final_from_hierarchy(
        full, a0t, a0v, a4v, denom_full, final_init_seed, final_fit_seed
    )
    full_m["final_replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(full_final)), Xv, yv
    )
    conditions["full_64"] = full_m

    direct = TinyRes(64, 32, final_init_seed)
    direct = fit_map(direct, a0t, a4t, 600, final_fit_seed)
    with torch.no_grad():
        direct_v = direct(a0v)
    direct_nmse = nmse(direct_v, a4v, denom_full)
    direct_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(direct)), Xv, yv)

    frozen_nmse = conditions["frozen"]["final_nmse_vs_original"]
    full_nmse = conditions["full_64"]["final_nmse_vs_original"]
    for name, metrics in conditions.items():
        metrics["difference_final_vs_frozen"] = metrics["final_nmse_vs_original"] - frozen_nmse
        metrics["ratio_final_over_full64"] = metrics["final_nmse_vs_original"] / full_nmse
        metrics["ratio_final_over_direct"] = metrics["final_nmse_vs_original"] / direct_nmse

    frontier_names = [f"self_{k}" for k in CAL_COUNTS] + ["self_all"]
    frontier_nmse = [conditions[n]["final_nmse_vs_original"] for n in frontier_names]

    return {
        "seed": seed,
        "status": "EXPLORATORY_OUTCOME",
        "test_evaluated": False,
        "teacher_val_acc": accuracy(teacher, Xv, yv),
        "train_span_sample_count": len(a0t),
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
        "projection": {"rng_seed": 20261410, "dimension": 16, "ambient_dimension": 64},
        "calibration_order_seed": seed + 190000,
        "conditions": conditions,
        "direct_original_single": {
            "final_nmse_vs_original": direct_nmse,
            "replacement_val_acc": direct_acc,
        },
        "frontier_names": frontier_names,
        "frontier_final_nmse": frontier_nmse,
        "frontier_monotone_nonincreasing": all(
            frontier_nmse[i + 1] <= frontier_nmse[i] for i in range(len(frontier_nmse) - 1)
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
