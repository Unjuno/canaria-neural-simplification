#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
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
    set_all_trainable,
    split_data,
    train_teacher,
)
from scripts.exploration.c12_self_anchored_sketches import adapt_anchored


FRESH_SEEDS = tuple(range(59400, 59416))
IMPLEMENTATION_VERIFICATION_SEED = 58400
CALIBRATION_SAMPLES = 192
CALIBRATION_INDEX_SEED = 20260903
GAUSSIAN_SIGMA = 0.04
FIT_UPDATES = 600


def sha256_tensor(x: torch.Tensor) -> str:
    a = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def sha256_int_array(x: np.ndarray) -> str:
    a = np.asarray(x, dtype=np.int64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def shifted_input(x: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(x.shape, generator=g, dtype=x.dtype) * GAUSSIAN_SIGMA
    return torch.clamp(x + noise, 0.0, 1.0)


def fixed_calibration_indices(n_train: int) -> np.ndarray:
    rng = np.random.default_rng(CALIBRATION_INDEX_SEED)
    return np.sort(rng.choice(n_train, size=CALIBRATION_SAMPLES, replace=False)).astype(np.int64)


def canonical_nested_qr(residual: torch.Tensor) -> torch.Tensor:
    # residual: [n_calibration, 64]. QR is applied to residual^T: [64, n].
    q, _r = torch.linalg.qr(residual.T, mode="reduced")
    # With n=192 and d=64, q is [64,64]. Canonicalize QR column signs.
    for j in range(q.shape[1]):
        col = q[:, j]
        pivot = int(torch.argmax(torch.abs(col)).item())
        if float(col[pivot]) < 0.0:
            q[:, j] = -col
    return q


def build_base_hierarchy(seed: int, a0t: torch.Tensor, a1t: torch.Tensor,
                         a2t: torch.Tensor, a3t: torch.Tensor,
                         a4t: torch.Tensor) -> tuple[Chain, dict]:
    locals4 = [
        fit_map(TinyRes(64, 8, seed + 101001), a0t, a1t, FIT_UPDATES, seed + 102001),
        fit_map(TinyRes(64, 8, seed + 101002), a1t, a2t, FIT_UPDATES, seed + 102002),
        fit_map(TinyRes(64, 8, seed + 101003), a2t, a3t, FIT_UPDATES, seed + 102003),
        fit_map(TinyRes(64, 8, seed + 101004), a3t, a4t, FIT_UPDATES, seed + 102004),
    ]

    pair12 = Chain([copy.deepcopy(locals4[0]), copy.deepcopy(locals4[1])])
    set_all_trainable(pair12, True)
    fit_map(pair12, a0t, a2t, FIT_UPDATES, seed + 110001)
    set_all_trainable(pair12, False)

    pair34 = Chain([copy.deepcopy(locals4[2]), copy.deepcopy(locals4[3])])
    set_all_trainable(pair34, True)
    fit_map(pair34, a2t, a4t, FIT_UPDATES, seed + 110002)
    set_all_trainable(pair34, False)

    with torch.no_grad():
        pair12_t = pair12(a0t).detach()
        pair34_t = pair34(a2t).detach()

    c12 = fit_map(TinyRes(64, 16, seed + 120001), a0t, pair12_t, FIT_UPDATES, seed + 121001)
    c34 = fit_map(TinyRes(64, 16, seed + 120002), a2t, pair34_t, FIT_UPDATES, seed + 121002)

    base = Chain([copy.deepcopy(c12), copy.deepcopy(c34)])
    set_all_trainable(base, False)

    budget = {
        "local_total_params": sum(count_params(m) for m in locals4),
        "level1_total_params": count_params(c12) + count_params(c34),
        "compiled_final_expected_params": 4096,
    }
    budget["exact_4096_each_level"] = bool(
        budget["local_total_params"] == budget["level1_total_params"] == 4096
    )
    return base, budget


def evaluate_condition(
    teacher,
    base_hierarchy: Chain,
    q: torch.Tensor,
    k: int,
    a0c: torch.Tensor,
    a4c: torch.Tensor,
    a0v_shift: torch.Tensor,
    a4v_shift: torch.Tensor,
    Xv_shift: torch.Tensor,
    yv: torch.Tensor,
    denom_shift: float,
    seed: int,
) -> dict:
    hierarchy = copy.deepcopy(base_hierarchy)
    with torch.no_grad():
        baseline_c = base_hierarchy(a0c).detach()
        residual_c = a4c - baseline_c
        p = q[:, :k]
        correction = (residual_c @ p) @ p.T
        hybrid_target = (baseline_c + correction).detach()

    adapt_anchored(hierarchy, a0c, hybrid_target, FIT_UPDATES, seed + 615000)
    set_all_trainable(hierarchy, False)

    final, metrics = compile_final_from_hierarchy(
        hierarchy,
        a0c,
        a0v_shift,
        a4v_shift,
        denom_shift,
        seed + 616000,
        seed + 617000,
    )
    metrics["final_replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(final)), Xv_shift, yv
    )
    metrics["dimension"] = int(k)
    return metrics


def run(seed: int, allow_verification_seed: bool = False) -> dict:
    if seed not in FRESH_SEEDS and not (allow_verification_seed and seed == IMPLEMENTATION_VERIFICATION_SEED):
        raise ValueError(
            f"seed {seed} is not in locked fresh cohort {FRESH_SEEDS[0]}-{FRESH_SEEDS[-1]} "
            f"and is not the verification seed {IMPLEMENTATION_VERIFICATION_SEED}"
        )

    Xt, yt, Xv, yv = split_data()
    teacher = train_teacher(seed, Xt, yt, 60)

    # Build the recursive hierarchy entirely from the clean training split.
    a0t, a1t, a2t, a3t, a4t = acts(teacher, Xt)
    base_hierarchy, budget = build_base_hierarchy(seed, a0t, a1t, a2t, a3t, a4t)

    # The protocol-fixed calibration subset is reused across dimensions and model seeds.
    cal_idx = fixed_calibration_indices(len(Xt))
    Xc_clean = Xt[torch.tensor(cal_idx, dtype=torch.long)]
    Xc_shift = shifted_input(Xc_clean, seed + 610100)
    Xv_shift = shifted_input(Xv, seed + 610200)

    ac = acts(teacher, Xc_shift)
    av_shift = acts(teacher, Xv_shift)
    a0c, a4c = ac[0], ac[-1]
    a0v_shift, a4v_shift = av_shift[0], av_shift[-1]
    denom_shift = float(((a4v_shift - a4v_shift.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    with torch.no_grad():
        baseline_c = base_hierarchy(a0c).detach()
        residual_c = a4c - baseline_c
    q = canonical_nested_qr(residual_c)

    # Frozen reference is informative; primary comparison is P4 vs P8.
    frozen_final, frozen_metrics = compile_final_from_hierarchy(
        copy.deepcopy(base_hierarchy),
        a0c,
        a0v_shift,
        a4v_shift,
        denom_shift,
        seed + 616000,
        seed + 617000,
    )
    frozen_metrics["final_replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(frozen_final)), Xv_shift, yv
    )

    p4 = evaluate_condition(
        teacher, base_hierarchy, q, 4, a0c, a4c, a0v_shift, a4v_shift,
        Xv_shift, yv, denom_shift, seed
    )
    p8 = evaluate_condition(
        teacher, base_hierarchy, q, 8, a0c, a4c, a0v_shift, a4v_shift,
        Xv_shift, yv, denom_shift, seed
    )

    required = [
        p4["final_replacement_val_acc"],
        p8["final_replacement_val_acc"],
        p4["final_nmse_vs_original"],
        p8["final_nmse_vs_original"],
    ]
    finite = all(math.isfinite(float(x)) for x in required)
    positive_nmse = float(p4["final_nmse_vs_original"]) > 0.0 and float(p8["final_nmse_vs_original"]) > 0.0
    eligible = bool(budget["exact_4096_each_level"] and finite and positive_nmse)

    return {
        "experiment": "C61R_GAUSSIAN_SHIFT_P4_VS_P8_PROSPECTIVE_REPLICATION",
        "status": "FRESH_SEED_OUTCOME" if seed in FRESH_SEEDS else "IMPLEMENTATION_VERIFICATION_OUTCOME",
        "seed": int(seed),
        "eligible": eligible,
        "test_evaluated": False,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "protocol": {
            "sigma": GAUSSIAN_SIGMA,
            "calibration_samples": CALIBRATION_SAMPLES,
            "calibration_index_seed": CALIBRATION_INDEX_SEED,
            "dimensions": [4, 8],
        },
        "provenance_hashes": {
            "calibration_indices_sha256": sha256_int_array(cal_idx),
            "shifted_calibration_tensor_sha256": sha256_tensor(Xc_shift),
            "shifted_validation_tensor_sha256": sha256_tensor(Xv_shift),
            "nested_q8_sha256": sha256_tensor(q[:, :8]),
        },
        "teacher_clean_validation_accuracy": accuracy(teacher, Xv, yv),
        "teacher_shifted_validation_accuracy": accuracy(teacher, Xv_shift, yv),
        "budget": budget,
        "frozen": frozen_metrics,
        "p4": p4,
        "p8": p8,
        "decision_row": {
            "seed": int(seed),
            "eligible": eligible,
            "p4_validation_accuracy": float(p4["final_replacement_val_acc"]),
            "p8_validation_accuracy": float(p8["final_replacement_val_acc"]),
            "p4_nmse": float(p4["final_nmse_vs_original"]),
            "p8_nmse": float(p8["final_nmse_vs_original"]),
            "frozen_nmse": float(frozen_metrics["final_nmse_vs_original"]),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-verification-seed", action="store_true")
    args = ap.parse_args()
    rec = run(args.seed, allow_verification_seed=args.allow_verification_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
