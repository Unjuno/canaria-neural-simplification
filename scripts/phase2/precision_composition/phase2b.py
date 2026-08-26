from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from common import PairReplacedNet, TinyRes, accuracy, acts, code_bits, datasets, fit_map, quantize_per_matrix, train_model

SEEDS = list(range(31100, 31108))
HS = [24, 32, 48, 64]
UPDATES = 600
BITS = 3


def run(seed: int):
    Xt, yt, Xv, yv, Xte, yte = datasets()
    teacher = train_model(seed, Xt, yt)
    train_acts, val_acts = acts(teacher, Xt), acts(teacher, Xv)
    target = val_acts[2]
    denom = float(((target - target.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    teacher_val = accuracy(teacher, Xv, yv)
    rows, selected_sep, selected_comp = [], None, None
    selected_sep_net = selected_comp_net = None
    for h in HS:
        budget = 256 * h
        r1 = fit_map(TinyRes(64, h, seed + 510000 + h), train_acts[0], train_acts[1], UPDATES, seed + 520000 + h)
        r2 = fit_map(TinyRes(64, h, seed + 530000 + h), train_acts[1], train_acts[2], UPDATES, seed + 540000 + h)
        comp = fit_map(TinyRes(64, 2 * h, seed + 550000 + h), train_acts[0], train_acts[2], UPDATES, seed + 560000 + h)
        qr1, s1 = quantize_per_matrix(r1, BITS)
        qr2, s2 = quantize_per_matrix(r2, BITS)
        qc, sc = quantize_per_matrix(comp, BITS)
        sepnet = PairReplacedNet(teacher, 0, r1=qr1, r2=qr2)
        compnet = PairReplacedNet(teacher, 0, comp=qc)
        with torch.no_grad():
            sep_nmse = float(F.mse_loss(qr2(qr1(val_acts[0])), target)) / denom
            comp_nmse = float(F.mse_loss(qc(val_acts[0]), target)) / denom
        sep_val = accuracy(sepnet, Xv, yv)
        comp_val = accuracy(compnet, Xv, yv)
        sep_pass = bool(sep_nmse <= 0.08 and sep_val >= teacher_val - 0.02)
        comp_pass = bool(comp_nmse <= 0.08 and comp_val >= teacher_val - 0.02)
        row = {
            "h": h, "budget_params": budget,
            "sep_nmse": sep_nmse, "comp_nmse": comp_nmse,
            "sep_val_acc": sep_val, "comp_val_acc": comp_val,
            "sep_pass": sep_pass, "comp_pass": comp_pass,
            "sep_code_bits": code_bits(budget, 3, s1 + s2),
            "comp_code_bits": code_bits(budget, 3, sc),
        }
        if sep_pass and selected_sep is None:
            selected_sep, selected_sep_net = dict(row), sepnet
        if comp_pass and selected_comp is None:
            selected_comp, selected_comp_net = dict(row), compnet
        rows.append(row)
    if selected_sep:
        selected_sep["sep_test_acc"] = accuracy(selected_sep_net, Xte, yte)
    if selected_comp:
        selected_comp["comp_test_acc"] = accuracy(selected_comp_net, Xte, yte)
    out = {
        "seed": seed, "teacher_val_acc": teacher_val,
        "teacher_test_acc": accuracy(teacher, Xte, yte),
        "grid": rows, "selected_sep": selected_sep, "selected_comp": selected_comp,
    }
    if selected_sep and selected_comp:
        out["log2_code_ratio"] = math.log2(selected_comp["comp_code_bits"] / selected_sep["sep_code_bits"])
        out["test_acc_diff"] = selected_comp["comp_test_acc"] - selected_sep["sep_test_acc"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.seed, result["selected_sep"]["budget_params"] if result["selected_sep"] else None, result["selected_comp"]["budget_params"] if result["selected_comp"] else None)


if __name__ == "__main__":
    main()
