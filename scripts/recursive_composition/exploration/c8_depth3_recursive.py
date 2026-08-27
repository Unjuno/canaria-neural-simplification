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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exploration.c6_hierarchical_recursive import (
    Chain,
    FullSpanReplacedNet,
    TinyRes,
    Net,
    accuracy,
    acts,
    count_params,
    fit_map,
    nmse,
    set_all_trainable,
)


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


def train_teacher8(seed, Xt, yt, epochs=60):
    model = Net(seed, depth=8)
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


def compile_from_frozen(module, width, init_seed, fit_seed, Xin_t, Xin_v, original_v, denom_original):
    with torch.no_grad():
        target_t = module(Xin_t).detach()
        target_v = module(Xin_v).detach()
    out = TinyRes(64, width, init_seed)
    out = fit_map(out, Xin_t, target_t, 600, fit_seed)
    with torch.no_grad():
        out_v = out(Xin_v)
    denom_target = float(((target_v - target_v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    return out, {
        "teacher_nmse_vs_original": nmse(target_v, original_v, denom_original),
        "compiled_nmse_vs_teacher": nmse(out_v, target_v, denom_target),
        "compiled_nmse_vs_original": nmse(out_v, original_v, denom_original),
    }


def run(seed):
    Xt, yt, Xv, yv = split_data()
    teacher = train_teacher8(seed, Xt, yt)
    at = acts(teacher, Xt)
    av = acts(teacher, Xv)
    assert len(at) == 9 and len(av) == 9
    denom_a8 = float(((av[8] - av[8].mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    local = []
    for i in range(8):
        local.append(
            fit_map(
                TinyRes(64, 4, seed + 101000 + i),
                at[i], at[i+1], 600, seed + 102000 + i,
            )
        )

    # Level 1: pair alignment is shared by both strict and realigned paths.
    level1 = []
    level1_pair_metrics = []
    for g, start in enumerate((0, 2, 4, 6)):
        pair = Chain([copy.deepcopy(local[start]), copy.deepcopy(local[start+1])])
        set_all_trainable(pair, True)
        fit_map(pair, at[start], at[start+2], 600, seed + 110000 + g)
        set_all_trainable(pair, False)
        unit, metrics = compile_from_frozen(
            pair, 8, seed + 120000 + g, seed + 121000 + g,
            at[start], av[start], av[start+2],
            float(((av[start+2] - av[start+2].mean(0, keepdim=True)) ** 2).mean()) + 1e-12,
        )
        level1.append(unit)
        level1_pair_metrics.append(metrics)

    with torch.no_grad():
        strict_l1_chain_v = Chain([copy.deepcopy(u) for u in level1])(av[0])
    strict_l1_full_nmse = nmse(strict_l1_chain_v, av[8], denom_a8)

    # Strict Level 2: no original 4-block re-alignment.
    strict_l2 = []
    strict_l2_metrics = []
    for g, start in enumerate((0, 4)):
        child = Chain([copy.deepcopy(level1[2*g]), copy.deepcopy(level1[2*g+1])])
        set_all_trainable(child, False)
        unit, metrics = compile_from_frozen(
            child, 16, seed + 130000 + g, seed + 131000 + g,
            at[start], av[start], av[start+4],
            float(((av[start+4] - av[start+4].mean(0, keepdim=True)) ** 2).mean()) + 1e-12,
        )
        strict_l2.append(unit)
        strict_l2_metrics.append(metrics)

    strict_l2_chain = Chain([copy.deepcopy(u) for u in strict_l2])
    set_all_trainable(strict_l2_chain, False)
    with torch.no_grad():
        strict_l2_chain_v = strict_l2_chain(av[0])
    strict_l2_full_nmse = nmse(strict_l2_chain_v, av[8], denom_a8)

    # Realigned Level 2: re-open each four-block boundary, align, freeze, then recursively compile.
    realign_l2 = []
    realign_l2_metrics = []
    for g, start in enumerate((0, 4)):
        child = Chain([copy.deepcopy(level1[2*g]), copy.deepcopy(level1[2*g+1])])
        set_all_trainable(child, True)
        fit_map(child, at[start], at[start+4], 600, seed + 140000 + g)
        set_all_trainable(child, False)
        unit, metrics = compile_from_frozen(
            child, 16, seed + 150000 + g, seed + 151000 + g,
            at[start], av[start], av[start+4],
            float(((av[start+4] - av[start+4].mean(0, keepdim=True)) ** 2).mean()) + 1e-12,
        )
        realign_l2.append(unit)
        realign_l2_metrics.append(metrics)

    realign_l2_chain = Chain([copy.deepcopy(u) for u in realign_l2])
    set_all_trainable(realign_l2_chain, False)
    with torch.no_grad():
        realign_l2_chain_v = realign_l2_chain(av[0])
    realign_l2_full_nmse = nmse(realign_l2_chain_v, av[8], denom_a8)

    final_init_seed = seed + 170000
    final_fit_seed = seed + 171000
    conditions = {}

    # Strict Level 3: no original 8-block re-alignment.
    strict_parent = copy.deepcopy(strict_l2_chain)
    strict_final, strict_m = compile_from_frozen(
        strict_parent, 32, final_init_seed, final_fit_seed,
        at[0], av[0], av[8], denom_a8,
    )
    strict_m["replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(strict_final)), Xv, yv
    )
    conditions["strict_depth3"] = strict_m

    # Realigned Level 3: re-open full boundary, align, freeze, then recursive compile.
    realign_parent = copy.deepcopy(realign_l2_chain)
    set_all_trainable(realign_parent, True)
    fit_map(realign_parent, at[0], at[8], 600, seed + 160000)
    set_all_trainable(realign_parent, False)
    realign_final, realign_m = compile_from_frozen(
        realign_parent, 32, final_init_seed, final_fit_seed,
        at[0], av[0], av[8], denom_a8,
    )
    realign_m["replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(realign_final)), Xv, yv
    )
    conditions["realign_each_level"] = realign_m

    # One-level recursive control from all 8 local candidates.
    single_parent = Chain([copy.deepcopy(u) for u in local])
    set_all_trainable(single_parent, True)
    fit_map(single_parent, at[0], at[8], 600, seed + 180000)
    set_all_trainable(single_parent, False)
    single_final, single_m = compile_from_frozen(
        single_parent, 32, final_init_seed, final_fit_seed,
        at[0], av[0], av[8], denom_a8,
    )
    single_m["replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(single_final)), Xv, yv
    )
    conditions["single_level_recursive"] = single_m

    # Direct original control.
    direct = TinyRes(64, 32, final_init_seed)
    direct = fit_map(direct, at[0], at[8], 600, final_fit_seed)
    with torch.no_grad():
        direct_v = direct(av[0])
    direct_nmse = nmse(direct_v, av[8], denom_a8)
    direct_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(direct)), Xv, yv)

    for rec in conditions.values():
        rec["ratio_final_nmse_over_direct"] = rec["compiled_nmse_vs_original"] / direct_nmse

    return {
        "seed": seed,
        "status": "EXPLORATORY_OUTCOME",
        "test_evaluated": False,
        "teacher_val_acc": accuracy(teacher, Xv, yv),
        "budget": {
            "level0_local_total": sum(count_params(u) for u in local),
            "level1_total": sum(count_params(u) for u in level1),
            "strict_level2_total": sum(count_params(u) for u in strict_l2),
            "realign_level2_total": sum(count_params(u) for u in realign_l2),
            "final_params": count_params(direct),
            "exact_4096_each_level": (
                sum(count_params(u) for u in local)
                == sum(count_params(u) for u in level1)
                == sum(count_params(u) for u in strict_l2)
                == sum(count_params(u) for u in realign_l2)
                == count_params(direct)
                == 4096
            ),
        },
        "depth_trajectory": {
            "strict_level1_full_chain_nmse": strict_l1_full_nmse,
            "strict_level2_full_chain_nmse": strict_l2_full_nmse,
            "strict_level3_final_nmse": conditions["strict_depth3"]["compiled_nmse_vs_original"],
            "realign_level1_full_chain_nmse": strict_l1_full_nmse,
            "realign_level2_full_chain_nmse": realign_l2_full_nmse,
            "realign_level3_final_nmse": conditions["realign_each_level"]["compiled_nmse_vs_original"],
        },
        "level1_pair_metrics": level1_pair_metrics,
        "strict_level2_metrics": strict_l2_metrics,
        "realign_level2_metrics": realign_l2_metrics,
        "conditions": conditions,
        "direct_original_single": {
            "final_nmse_vs_original": direct_nmse,
            "replacement_val_acc": direct_acc,
        },
        "derived": {
            "D_realign_minus_strict": conditions["realign_each_level"]["compiled_nmse_vs_original"]
            - conditions["strict_depth3"]["compiled_nmse_vs_original"],
            "D_realign_minus_single": conditions["realign_each_level"]["compiled_nmse_vs_original"]
            - conditions["single_level_recursive"]["compiled_nmse_vs_original"],
            "R_realign_over_direct": conditions["realign_each_level"]["compiled_nmse_vs_original"] / direct_nmse,
            "R_strict_over_direct": conditions["strict_depth3"]["compiled_nmse_vs_original"] / direct_nmse,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    rec = run(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2), encoding='utf-8')
    print(json.dumps(rec, indent=2))


if __name__ == '__main__':
    main()
