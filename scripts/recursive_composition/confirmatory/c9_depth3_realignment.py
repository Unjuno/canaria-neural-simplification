from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exploration.c6_hierarchical_recursive import (
    Chain,
    FullSpanReplacedNet,
    TinyRes,
    accuracy,
    acts,
    count_params,
    fit_map,
    nmse,
    set_all_trainable,
)
from scripts.exploration.c8_depth3_recursive import train_teacher8, compile_from_frozen


def split_data_and_test_index():
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
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
        test_idx.tolist(),
    )


def materialize_test(test_idx):
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    ix = np.asarray(test_idx, dtype=np.int64)
    return torch.tensor(X[ix]), torch.tensor(y[ix], dtype=torch.long)


def run(seed):
    Xt, yt, Xv, yv, test_idx = split_data_and_test_index()
    teacher = train_teacher8(seed, Xt, yt, 60)
    at = acts(teacher, Xt)
    av = acts(teacher, Xv)
    denom_a8 = float(((av[8] - av[8].mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    local = []
    for i in range(8):
        local.append(
            fit_map(
                TinyRes(64, 4, seed + 101000 + i),
                at[i], at[i+1], 600, seed + 102000 + i,
            )
        )

    level1 = []
    for g, start in enumerate((0, 2, 4, 6)):
        pair = Chain([copy.deepcopy(local[start]), copy.deepcopy(local[start+1])])
        set_all_trainable(pair, True)
        fit_map(pair, at[start], at[start+2], 600, seed + 110000 + g)
        set_all_trainable(pair, False)
        unit, _ = compile_from_frozen(
            pair, 8, seed + 120000 + g, seed + 121000 + g,
            at[start], av[start], av[start+2],
            float(((av[start+2] - av[start+2].mean(0, keepdim=True)) ** 2).mean()) + 1e-12,
        )
        level1.append(unit)

    with torch.no_grad():
        level1_full_v = Chain([copy.deepcopy(u) for u in level1])(av[0])
    level1_full_nmse = nmse(level1_full_v, av[8], denom_a8)

    strict_l2 = []
    for g, start in enumerate((0, 4)):
        child = Chain([copy.deepcopy(level1[2*g]), copy.deepcopy(level1[2*g+1])])
        set_all_trainable(child, False)
        unit, _ = compile_from_frozen(
            child, 16, seed + 130000 + g, seed + 131000 + g,
            at[start], av[start], av[start+4],
            float(((av[start+4] - av[start+4].mean(0, keepdim=True)) ** 2).mean()) + 1e-12,
        )
        strict_l2.append(unit)
    strict_l2_chain = Chain([copy.deepcopy(u) for u in strict_l2])
    set_all_trainable(strict_l2_chain, False)
    with torch.no_grad():
        strict_l2_v = strict_l2_chain(av[0])
    strict_l2_nmse = nmse(strict_l2_v, av[8], denom_a8)

    realign_l2 = []
    for g, start in enumerate((0, 4)):
        child = Chain([copy.deepcopy(level1[2*g]), copy.deepcopy(level1[2*g+1])])
        set_all_trainable(child, True)
        fit_map(child, at[start], at[start+4], 600, seed + 140000 + g)
        set_all_trainable(child, False)
        unit, _ = compile_from_frozen(
            child, 16, seed + 150000 + g, seed + 151000 + g,
            at[start], av[start], av[start+4],
            float(((av[start+4] - av[start+4].mean(0, keepdim=True)) ** 2).mean()) + 1e-12,
        )
        realign_l2.append(unit)
    realign_l2_chain = Chain([copy.deepcopy(u) for u in realign_l2])
    set_all_trainable(realign_l2_chain, False)
    with torch.no_grad():
        realign_l2_v = realign_l2_chain(av[0])
    realign_l2_nmse = nmse(realign_l2_v, av[8], denom_a8)

    final_init_seed = seed + 170000
    final_fit_seed = seed + 171000
    conditions = {}
    fitted = {}

    strict_final, strict_m = compile_from_frozen(
        strict_l2_chain, 32, final_init_seed, final_fit_seed,
        at[0], av[0], av[8], denom_a8,
    )
    strict_m['replacement_val_acc'] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(strict_final)), Xv, yv
    )
    conditions['strict_depth3'] = strict_m
    fitted['strict_depth3'] = copy.deepcopy(strict_final)

    realign_parent = copy.deepcopy(realign_l2_chain)
    set_all_trainable(realign_parent, True)
    fit_map(realign_parent, at[0], at[8], 600, seed + 160000)
    set_all_trainable(realign_parent, False)
    realign_final, realign_m = compile_from_frozen(
        realign_parent, 32, final_init_seed, final_fit_seed,
        at[0], av[0], av[8], denom_a8,
    )
    realign_m['replacement_val_acc'] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(realign_final)), Xv, yv
    )
    conditions['realign_each_level'] = realign_m
    fitted['realign_each_level'] = copy.deepcopy(realign_final)

    single_parent = Chain([copy.deepcopy(u) for u in local])
    set_all_trainable(single_parent, True)
    fit_map(single_parent, at[0], at[8], 600, seed + 180000)
    set_all_trainable(single_parent, False)
    single_final, single_m = compile_from_frozen(
        single_parent, 32, final_init_seed, final_fit_seed,
        at[0], av[0], av[8], denom_a8,
    )
    single_m['replacement_val_acc'] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(single_final)), Xv, yv
    )
    conditions['single_level_recursive'] = single_m
    fitted['single_level_recursive'] = copy.deepcopy(single_final)

    direct = TinyRes(64, 32, final_init_seed)
    direct = fit_map(direct, at[0], at[8], 600, final_fit_seed)
    with torch.no_grad():
        direct_v = direct(av[0])
    direct_nmse = nmse(direct_v, av[8], denom_a8)
    direct_val_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(direct)), Xv, yv)

    validation = {
        'D_strict': conditions['realign_each_level']['compiled_nmse_vs_original']
        - conditions['strict_depth3']['compiled_nmse_vs_original'],
        'R_realign': conditions['realign_each_level']['compiled_nmse_vs_original'] / direct_nmse,
        'D_single': conditions['realign_each_level']['compiled_nmse_vs_original']
        - conditions['single_level_recursive']['compiled_nmse_vs_original'],
        'D_level2': realign_l2_nmse - strict_l2_nmse,
        'val_acc_diff': conditions['realign_each_level']['replacement_val_acc'] - direct_val_acc,
    }

    Xte, yte = materialize_test(test_idx)
    teacher_test_acc = accuracy(teacher, Xte, yte)
    for name, model in fitted.items():
        conditions[name]['replacement_test_acc'] = accuracy(
            FullSpanReplacedNet(teacher, copy.deepcopy(model)), Xte, yte
        )
    direct_test_acc = accuracy(FullSpanReplacedNet(teacher, copy.deepcopy(direct)), Xte, yte)
    validation['test_acc_diff'] = (
        conditions['realign_each_level']['replacement_test_acc'] - direct_test_acc
    )

    return {
        'seed': seed,
        'status': 'CONFIRMATORY_SEED_OUTCOME',
        'test_evaluated_after_validation_only': True,
        'teacher_val_acc': accuracy(teacher, Xv, yv),
        'teacher_test_acc': teacher_test_acc,
        'budget': {
            'level0_total': sum(count_params(u) for u in local),
            'level1_total': sum(count_params(u) for u in level1),
            'strict_level2_total': sum(count_params(u) for u in strict_l2),
            'realign_level2_total': sum(count_params(u) for u in realign_l2),
            'final_total': count_params(direct),
            'exact_4096_each_level': (
                sum(count_params(u) for u in local)
                == sum(count_params(u) for u in level1)
                == sum(count_params(u) for u in strict_l2)
                == sum(count_params(u) for u in realign_l2)
                == count_params(direct)
                == 4096
            ),
        },
        'depth_trajectory': {
            'level1_full_nmse': level1_full_nmse,
            'strict_level2_full_nmse': strict_l2_nmse,
            'realign_level2_full_nmse': realign_l2_nmse,
            'strict_level3_final_nmse': conditions['strict_depth3']['compiled_nmse_vs_original'],
            'realign_level3_final_nmse': conditions['realign_each_level']['compiled_nmse_vs_original'],
        },
        'conditions': conditions,
        'direct_original_single': {
            'final_nmse_vs_original': direct_nmse,
            'replacement_val_acc': direct_val_acc,
            'replacement_test_acc': direct_test_acc,
        },
        'derived': validation,
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
