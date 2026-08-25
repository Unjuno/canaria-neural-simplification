from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    PairReplacedNet, TinyRes, accuracy, acts, code_bits, datasets, fit_map,
    quantize_per_matrix, train_model,
)

SEEDS = list(range(31000, 31008))
BITS = [32, 12, 8, 6, 4, 3]
HS = [2, 4, 6, 8, 12, 16, 24]
UPDATES = 600


def one_seed(seed: int):
    Xt, yt, Xv, yv, Xte, yte = datasets()
    teacher = train_model(seed, Xt, yt)
    train_acts, val_acts = acts(teacher, Xt), acts(teacher, Xv)
    target = val_acts[2]
    denom = float(((target - target.mean(0, keepdim=True)) ** 2).mean()) + 1e-12
    teacher_val = accuracy(teacher, Xv, yv)
    teacher_test = accuracy(teacher, Xte, yte)

    trained = {}
    for h in HS:
        r1 = fit_map(TinyRes(64, h, seed + 410000 + h), train_acts[0], train_acts[1], UPDATES, seed + 420000 + h)
        r2 = fit_map(TinyRes(64, h, seed + 430000 + h), train_acts[1], train_acts[2], UPDATES, seed + 440000 + h)
        comp = fit_map(TinyRes(64, 2 * h, seed + 450000 + h), train_acts[0], train_acts[2], UPDATES, seed + 460000 + h)
        trained[h] = (r1, r2, comp)

    by_bits = {}
    for bits in BITS:
        rows, selected_sep, selected_comp = [], None, None
        selected_sep_net = selected_comp_net = None
        for h in HS:
            budget = 256 * h
            r1, r2, comp = trained[h]
            qr1, s1 = quantize_per_matrix(r1, bits)
            qr2, s2 = quantize_per_matrix(r2, bits)
            qc, sc = quantize_per_matrix(comp, bits)
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
                "sep_code_bits": code_bits(budget, bits, s1 + s2),
                "comp_code_bits": code_bits(budget, bits, sc),
                "sep_scales": s1 + s2, "comp_scales": sc,
            }
            if sep_pass and selected_sep is None:
                selected_sep, selected_sep_net = dict(row), sepnet
            if comp_pass and selected_comp is None:
                selected_comp, selected_comp_net = dict(row), compnet
            rows.append(row)
        if selected_sep is not None:
            selected_sep["sep_test_acc"] = accuracy(selected_sep_net, Xte, yte)
        if selected_comp is not None:
            selected_comp["comp_test_acc"] = accuracy(selected_comp_net, Xte, yte)
        rec = {"grid": rows, "selected_sep": selected_sep, "selected_comp": selected_comp}
        if selected_sep and selected_comp:
            rec["log2_param_ratio"] = math.log2(selected_comp["budget_params"] / selected_sep["budget_params"])
            rec["log2_code_ratio"] = math.log2(selected_comp["comp_code_bits"] / selected_sep["sep_code_bits"])
            rec["test_acc_diff_comp_minus_sep"] = selected_comp["comp_test_acc"] - selected_sep["sep_test_acc"]
        by_bits[str(bits)] = rec
    return {"seed": seed, "teacher_val_acc": teacher_val, "teacher_test_acc": teacher_test, "bits": by_bits}


def boot_ci(values, nboot=20000, seed=260826):
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.empty(nboot)
    for i in range(nboot):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(results):
    out = {}
    for bits in BITS:
        records = [r["bits"][str(bits)] for r in results]
        valid = [r for r in records if "log2_code_ratio" in r]
        ratios = [r["log2_code_ratio"] for r in valid]
        param_ratios = [r["log2_param_ratio"] for r in valid]
        test_diffs = [r["test_acc_diff_comp_minus_sep"] for r in valid]
        wins = sum(r["selected_comp"]["comp_code_bits"] < r["selected_sep"]["sep_code_bits"] for r in valid)
        out[str(bits)] = {
            "n_valid": len(valid), "wins_code": wins,
            "mean_log2_code_ratio": float(np.mean(ratios)) if ratios else None,
            "bootstrap95_log2_code_ratio": boot_ci(ratios, seed=260826 + bits) if ratios else None,
            "geometric_code_ratio": float(2 ** np.mean(ratios)) if ratios else None,
            "mean_log2_param_ratio": float(np.mean(param_ratios)) if param_ratios else None,
            "mean_test_acc_diff": float(np.mean(test_diffs)) if test_diffs else None,
            "bootstrap95_test_acc_diff": boot_ci(test_diffs, seed=270826 + bits) if test_diffs else None,
            "sep_budgets": [r["selected_sep"]["budget_params"] for r in valid],
            "comp_budgets": [r["selected_comp"]["budget_params"] for r in valid],
        }
    p = out["4"]
    if p["n_valid"] == 8 and p["wins_code"] >= 7 and p["bootstrap95_log2_code_ratio"][1] < 0:
        decision = "PASS"
    elif p["n_valid"] == 8 and (p["wins_code"] <= 1 or p["bootstrap95_log2_code_ratio"][0] >= 0):
        decision = "FAIL"
    else:
        decision = "UNCERTAIN"
    return {"primary_4bit_decision": decision, "by_bits": out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-dir", type=Path, default=Path("phase2a_out"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed] if args.seed is not None else SEEDS
    results = []
    for seed in seeds:
        result = one_seed(seed)
        results.append(result)
        (args.out_dir / f"seed_{seed}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(seed, result["bits"]["4"].get("log2_code_ratio"))
    if args.seed is None:
        (args.out_dir / "summary.json").write_text(json.dumps(summarize(results), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
