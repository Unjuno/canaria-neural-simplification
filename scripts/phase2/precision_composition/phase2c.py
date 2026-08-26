from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common import PairReplacedNet, TinyRes, accuracy, acts, datasets, fit_map, quantize_per_matrix, quantize_rowwise, train_model

BITS = 3
H = 64
BUDGET = 256 * H
UPDATES = 600


def run(seed: int):
    Xt, yt, Xv, yv, Xte, yte = datasets()
    teacher = train_model(seed, Xt, yt)
    train_acts, val_acts = acts(teacher, Xt), acts(teacher, Xv)
    target = val_acts[2]
    denom = float(((target - target.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    teacher_val = accuracy(teacher, Xv, yv)
    r1 = fit_map(TinyRes(64, H, seed + 610064), train_acts[0], train_acts[1], UPDATES, seed + 620064)
    r2 = fit_map(TinyRes(64, H, seed + 630064), train_acts[1], train_acts[2], UPDATES, seed + 640064)
    comp = fit_map(TinyRes(64, 2 * H, seed + 650064), train_acts[0], train_acts[2], UPDATES, seed + 660064)
    out = {
        "seed": seed, "teacher_val_acc": teacher_val,
        "teacher_test_acc": accuracy(teacher, Xte, yte), "conditions": {},
    }
    for name, qfn in [("per_matrix", quantize_per_matrix), ("rowwise", quantize_rowwise)]:
        qr1, s1 = qfn(r1, BITS)
        qr2, s2 = qfn(r2, BITS)
        qc, sc = qfn(comp, BITS)
        sepnet = PairReplacedNet(teacher, 0, r1=qr1, r2=qr2)
        compnet = PairReplacedNet(teacher, 0, comp=qc)
        with torch.no_grad():
            sep_nmse = float(F.mse_loss(qr2(qr1(val_acts[0])), target)) / denom
            comp_nmse = float(F.mse_loss(qc(val_acts[0]), target)) / denom
        sep_val = accuracy(sepnet, Xv, yv)
        comp_val = accuracy(compnet, Xv, yv)
        sep_pass = bool(sep_nmse <= 0.08 and sep_val >= teacher_val - 0.02)
        comp_pass = bool(comp_nmse <= 0.08 and comp_val >= teacher_val - 0.02)
        out["conditions"][name] = {
            "sep_nmse": sep_nmse, "comp_nmse": comp_nmse,
            "sep_val_acc": sep_val, "comp_val_acc": comp_val,
            "sep_pass": sep_pass, "comp_pass": comp_pass,
            "sep_scale_count": s1 + s2, "comp_scale_count": sc,
            "sep_code_bits": BUDGET * 3 + (s1 + s2) * 16,
            "comp_code_bits": BUDGET * 3 + sc * 16,
            "sep_test_acc": accuracy(sepnet, Xte, yte) if sep_pass else None,
            "comp_test_acc": accuracy(compnet, Xte, yte) if comp_pass else None,
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.seed, result["conditions"]["per_matrix"]["comp_pass"], result["conditions"]["rowwise"]["comp_pass"])


if __name__ == "__main__":
    main()
