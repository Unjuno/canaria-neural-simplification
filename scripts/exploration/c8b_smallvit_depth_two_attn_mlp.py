from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.exploration.c8_smallvit_depth_two_attention import (
    Cluster,
    SmallViT,
    WholeReplacedViT,
    accuracy,
    collect_acts,
    count_params,
    data_split,
    fit_map,
    nmse,
    set_all_trainable,
    set_seed,
    train_teacher,
)


class TinyAttnMLPRes(nn.Module):
    def __init__(self, d: int, width: int, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.d = d
        self.width = width
        self.q = nn.Linear(d, width, bias=False)
        self.k = nn.Linear(d, width, bias=False)
        self.v = nn.Linear(d, width, bias=False)
        self.o = nn.Linear(width, d, bias=False)
        self.ff1 = nn.Linear(d, width, bias=False)
        self.ff2 = nn.Linear(width, d, bias=False)

    def forward(self, x):
        z = F.layer_norm(x, (self.d,))
        q = self.q(z); k = self.k(z); v = self.v(z)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.width)
        x = x + self.o(torch.matmul(torch.softmax(score, dim=-1), v))
        z = F.layer_norm(x, (self.d,))
        return x + self.ff2(F.gelu(self.ff1(z)))


def final_fit(seed, target_t, target_v, a0t, a0v, a4v, original_denom, teacher, va):
    single = TinyAttnMLPRes(32, 128, seed + 160001)
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
    a0v, _a1v, a2v, _a3v, a4v = av
    original_denom = float(((a4v - a4v.mean(dim=(0, 1), keepdim=True)) ** 2).mean()) + 1e-12

    locals4 = [
        fit_map(TinyAttnMLPRes(32,32,seed+101001),a0t,a1t,600,seed+102001),
        fit_map(TinyAttnMLPRes(32,32,seed+101002),a1t,a2t,600,seed+102002),
        fit_map(TinyAttnMLPRes(32,32,seed+101003),a2t,a3t,600,seed+102003),
        fit_map(TinyAttnMLPRes(32,32,seed+101004),a3t,a4t,600,seed+102004),
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

    c12 = fit_map(TinyAttnMLPRes(32,64,seed+130001), a0t,p12t,600,seed+140001)
    c34 = fit_map(TinyAttnMLPRes(32,64,seed+130002), a2t,p34t,600,seed+140002)
    with torch.no_grad():
        pre_t,pre_v = c34(c12(a0t)).detach(), c34(c12(a0v)).detach()

    l2 = Cluster([copy.deepcopy(c12), copy.deepcopy(c34)])
    set_all_trainable(l2, True)
    l2 = fit_map(l2, a0t, lower_t, 600, seed+150001)
    set_all_trainable(l2, False)
    with torch.no_grad():
        post_t,post_v = l2(a0t).detach(), l2(a0v).detach()

    finals = {}
    _, finals["hierarchical_no_level2_adapt"] = final_fit(seed,pre_t,pre_v,a0t,a0v,a4v,original_denom,teacher,va)
    _, finals["hierarchical_level2_joint_adapt"] = final_fit(seed,post_t,post_v,a0t,a0v,a4v,original_denom,teacher,va)
    _, finals["flat_lower_ir_single"] = final_fit(seed,lower_t,lower_v,a0t,a0v,a4v,original_denom,teacher,va)
    _, finals["direct_original_single"] = final_fit(seed,a4t,a4v,a0t,a0v,a4v,original_denom,teacher,va)

    lower_denom = float(((lower_v-lower_v.mean(dim=(0,1),keepdim=True))**2).mean()) + 1e-12
    with torch.no_grad(): post_v_now = l2(a0v)
    joint = finals["hierarchical_level2_joint_adapt"]; no_adapt = finals["hierarchical_no_level2_adapt"]; flat = finals["flat_lower_ir_single"]; direct = finals["direct_original_single"]
    result.update({
        "fit_examples": 512,
        "budget": {"level0_total_params": sum(count_params(m) for m in locals4), "level1_total_params": count_params(c12)+count_params(c34), "level2_params": count_params(TinyAttnMLPRes(32,128,0)), "exact_24576_match": sum(count_params(m) for m in locals4) == count_params(c12)+count_params(c34) == count_params(TinyAttnMLPRes(32,128,0)) == 24576},
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
