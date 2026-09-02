#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gaussian_shift_interface.run_c61r_seed import (
    build_base_hierarchy,
    canonical_nested_qr,
    evaluate_condition,
    fixed_calibration_indices,
    sha256_int_array,
    sha256_tensor,
    shifted_input,
)
from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import (
    FullSpanReplacedNet,
    accuracy,
    acts,
    compile_final_from_hierarchy,
    split_data,
    train_teacher,
)

FRESH_SEEDS = tuple(range(60400, 60416))
IMPLEMENTATION_VERIFICATION_SEED = 60300
CALIBRATION_SAMPLES = 192
GAUSSIAN_SIGMA = 0.04
RANDOM_BASIS_SEED = 20260904
DIMENSIONS = (2, 4)


def canonicalize_columns(q: torch.Tensor) -> torch.Tensor:
    q = q.clone()
    for j in range(q.shape[1]):
        col = q[:, j]
        pivot = int(torch.argmax(torch.abs(col)).item())
        if float(col[pivot]) < 0.0:
            q[:, j] = -col
    return q


def fixed_random_basis() -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(RANDOM_BASIS_SEED)
    a = torch.randn((64, 64), generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(a, mode="reduced")
    return canonicalize_columns(q)


def residual_svd_basis(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _u, s, vh = torch.linalg.svd(residual, full_matrices=False)
    v = canonicalize_columns(vh.T)
    return v, s


def energy_capture(residual: torch.Tensor, basis: torch.Tensor, k: int) -> float:
    total = torch.sum(residual * residual)
    if float(total) <= 0.0:
        return 0.0
    coeff = residual @ basis[:, :k]
    return float(torch.sum(coeff * coeff) / total)


def subspace_alignment(a: torch.Tensor, b: torch.Tensor, k: int) -> dict:
    cosines = torch.linalg.svdvals(a[:, :k].T @ b[:, :k])
    cos2 = cosines * cosines
    return {
        "mean_squared_principal_cosine": float(cos2.mean()),
        "min_principal_cosine": float(cosines.min()),
        "principal_cosines": [float(x) for x in cosines],
    }


def spectrum_diagnostics(s: torch.Tensor) -> dict:
    power = s * s
    total = torch.sum(power)
    p = power / total
    nz = p[p > 0]
    entropy_effective_rank = torch.exp(-torch.sum(nz * torch.log(nz)))
    stable_rank = total / torch.max(power)
    out = {
        "entropy_effective_rank": float(entropy_effective_rank),
        "stable_rank": float(stable_rank),
    }
    for k in (2, 4, 8):
        out[f"optimal_svd_energy_fraction_k{k}"] = float(torch.sum(power[:k]) / total)
    return out


def run(seed: int, allow_verification_seed: bool = False) -> dict:
    torch.set_num_threads(1)
    if seed not in FRESH_SEEDS and not (
        allow_verification_seed and seed == IMPLEMENTATION_VERIFICATION_SEED
    ):
        raise ValueError(
            f"seed {seed} is not in locked fresh cohort {FRESH_SEEDS[0]}-{FRESH_SEEDS[-1]} "
            f"and is not verification seed {IMPLEMENTATION_VERIFICATION_SEED}"
        )

    Xt, yt, Xv, yv = split_data()
    teacher = train_teacher(seed, Xt, yt, 60)
    a0t, a1t, a2t, a3t, a4t = acts(teacher, Xt)
    base_hierarchy, budget = build_base_hierarchy(seed, a0t, a1t, a2t, a3t, a4t)

    cal_idx = fixed_calibration_indices(len(Xt))
    cal_t = torch.tensor(cal_idx, dtype=torch.long)
    Xc_shift = shifted_input(Xt[cal_t], seed + 620100)
    Xv_shift = shifted_input(Xv, seed + 620200)

    ac = acts(teacher, Xc_shift)
    av = acts(teacher, Xv_shift)
    a0c, a4c = ac[0], ac[-1]
    a0v, a4v = av[0], av[-1]
    denom = float(((a4v - a4v.mean(0, keepdim=True)) ** 2).mean()) + 1e-12

    with torch.no_grad():
        baseline_c = base_hierarchy(a0c).detach()
        baseline_v = base_hierarchy(a0v).detach()
        residual_c = a4c - baseline_c
        residual_v = a4v - baseline_v

    qr = canonical_nested_qr(residual_c)
    svd, singular_values = residual_svd_basis(residual_c)
    random = fixed_random_basis()
    bases = {"qr": qr, "svd": svd, "random": random}

    frozen_final, frozen_metrics = compile_final_from_hierarchy(
        copy.deepcopy(base_hierarchy), a0c, a0v, a4v, denom,
        seed + 616000, seed + 617000
    )
    frozen_metrics["final_replacement_val_acc"] = accuracy(
        FullSpanReplacedNet(teacher, copy.deepcopy(frozen_final)), Xv_shift, yv
    )

    conditions: dict[str, dict] = {}
    for basis_name, basis in bases.items():
        for k in DIMENSIONS:
            key = f"{basis_name}_p{k}"
            conditions[key] = evaluate_condition(
                teacher, base_hierarchy, basis, k,
                a0c, a4c, a0v, a4v, Xv_shift, yv, denom, seed
            )

    diagnostics = {
        "spectrum": spectrum_diagnostics(singular_values),
        "energy_capture": {},
        "qr_to_svd_alignment": {},
    }
    for basis_name, basis in bases.items():
        diagnostics["energy_capture"][basis_name] = {}
        for k in DIMENSIONS:
            diagnostics["energy_capture"][basis_name][f"k{k}"] = {
                "calibration": energy_capture(residual_c, basis, k),
                "shifted_validation": energy_capture(residual_v, basis, k),
            }
    for k in DIMENSIONS:
        diagnostics["qr_to_svd_alignment"][f"k{k}"] = subspace_alignment(qr, svd, k)

    required = []
    for rec in conditions.values():
        required.extend([
            rec["final_replacement_val_acc"],
            rec["final_nmse_vs_original"],
        ])
    finite = all(math.isfinite(float(x)) for x in required)
    positive_nmse = all(float(rec["final_nmse_vs_original"]) > 0.0 for rec in conditions.values())
    eligible = bool(budget["exact_4096_each_level"] and finite and positive_nmse)

    decision_row = {
        "seed": int(seed),
        "eligible": eligible,
        "frozen_nmse": float(frozen_metrics["final_nmse_vs_original"]),
        "conditions": {
            key: {
                "validation_accuracy": float(rec["final_replacement_val_acc"]),
                "nmse": float(rec["final_nmse_vs_original"]),
            }
            for key, rec in conditions.items()
        },
        "diagnostics": diagnostics,
    }

    return {
        "experiment": "C62R_P2_FRONTIER_AND_BASIS_MECHANISM_EXPLORATION",
        "evidence_class": "PROSPECTIVE_EXPLORATORY",
        "status": "FRESH_SEED_OUTCOME" if seed in FRESH_SEEDS else "IMPLEMENTATION_VERIFICATION_OUTCOME",
        "seed": int(seed),
        "eligible": eligible,
        "test_evaluated": False,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "protocol": {
            "sigma": GAUSSIAN_SIGMA,
            "calibration_samples": CALIBRATION_SAMPLES,
            "dimensions": list(DIMENSIONS),
            "basis_families": list(bases),
            "random_basis_seed": RANDOM_BASIS_SEED,
        },
        "provenance_hashes": {
            "calibration_indices_sha256": sha256_int_array(cal_idx),
            "shifted_calibration_tensor_sha256": sha256_tensor(Xc_shift),
            "shifted_validation_tensor_sha256": sha256_tensor(Xv_shift),
            "qr_p4_sha256": sha256_tensor(qr[:, :4]),
            "svd_p4_sha256": sha256_tensor(svd[:, :4]),
            "random_p4_sha256": sha256_tensor(random[:, :4]),
        },
        "teacher_clean_validation_accuracy": accuracy(teacher, Xv, yv),
        "teacher_shifted_validation_accuracy": accuracy(teacher, Xv_shift, yv),
        "budget": budget,
        "frozen": frozen_metrics,
        "conditions": conditions,
        "diagnostics": diagnostics,
        "decision_row": decision_row,
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
