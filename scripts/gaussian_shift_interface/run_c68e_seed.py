#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gaussian_shift_interface.run_c61r_seed import sha256_tensor
from scripts.recursive_composition.exploration.c10_boundary_signal_ablation import Net, split_data

FRESH_SEEDS = tuple(range(66400, 66416))
IMPLEMENTATION_VERIFICATION_SEED = 66300
EPOCHS = 60
BATCH_SIZE = 64
TARGET_SIGMA = 0.36
AUGMENT_FRACTION = 0.5
PERM_OFFSET = 999
AUGMENT_OFFSET = 680100
VAL_EPS_OFFSET = 680200


def state_dict_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        t = tensor.detach().cpu().contiguous()
        h.update(str(t.dtype).encode("ascii"))
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(t.numpy().tobytes(order="C"))
    return h.hexdigest()


def model_accuracy(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return float((model(x).argmax(-1) == y).float().mean())


def train_paired_teachers(seed: int, xt: torch.Tensor, yt: torch.Tensor):
    clean = Net(seed)
    augmented = Net(seed)
    clean_init_hash = state_dict_sha256(clean)
    augmented_init_hash = state_dict_sha256(augmented)
    if clean_init_hash != augmented_init_hash:
        raise AssertionError("paired teacher initializations differ")

    clean_opt = torch.optim.AdamW(clean.parameters(), lr=2e-3, weight_decay=1e-4)
    aug_opt = torch.optim.AdamW(augmented.parameters(), lr=2e-3, weight_decay=1e-4)
    perm_gen = torch.Generator().manual_seed(seed + PERM_OFFSET)
    aug_gen = torch.Generator().manual_seed(seed + AUGMENT_OFFSET)
    selected_total = 0
    examples_total = 0

    for _ in range(EPOCHS):
        perm = torch.randperm(len(xt), generator=perm_gen)
        for i in range(0, len(xt), BATCH_SIZE):
            ix = perm[i:i + BATCH_SIZE]
            xb = xt[ix]
            yb = yt[ix]

            clean_opt.zero_grad()
            clean_loss = F.cross_entropy(clean(xb), yb)
            clean_loss.backward()
            clean_opt.step()

            mask = torch.rand((len(xb),), generator=aug_gen) < AUGMENT_FRACTION
            eps = torch.randn(xb.shape, generator=aug_gen, dtype=xb.dtype)
            x_aug = torch.clamp(
                xb + mask.to(xb.dtype).unsqueeze(1) * TARGET_SIGMA * eps,
                0.0,
                1.0,
            )
            selected_total += int(mask.sum())
            examples_total += int(len(mask))

            aug_opt.zero_grad()
            aug_loss = F.cross_entropy(augmented(x_aug), yb)
            aug_loss.backward()
            aug_opt.step()

    return clean, augmented, {
        "paired_initial_state_sha256": clean_init_hash,
        "augmentation_selected_examples": selected_total,
        "augmentation_total_examples": examples_total,
        "realized_augmentation_fraction": float(selected_total / max(examples_total, 1)),
    }


def shifted_validation(xv: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed + VAL_EPS_OFFSET)
    eps = torch.randn(xv.shape, generator=gen, dtype=xv.dtype)
    shifted = torch.clamp(xv + TARGET_SIGMA * eps, 0.0, 1.0)
    return shifted, eps


def run(seed: int, allow_verification_seed: bool = False) -> dict:
    torch.set_num_threads(1)
    if seed not in FRESH_SEEDS and not (
        allow_verification_seed and seed == IMPLEMENTATION_VERIFICATION_SEED
    ):
        raise ValueError(
            f"seed {seed} is not in locked fresh cohort {FRESH_SEEDS[0]}-{FRESH_SEEDS[-1]} "
            f"and is not verification seed {IMPLEMENTATION_VERIFICATION_SEED}"
        )

    xt, yt, xv, yv = split_data()
    clean, augmented, training_prov = train_paired_teachers(seed, xt, yt)
    xv_shift, val_eps = shifted_validation(xv, seed)

    clean_clean = model_accuracy(clean, xv, yv)
    clean_shift = model_accuracy(clean, xv_shift, yv)
    aug_clean = model_accuracy(augmented, xv, yv)
    aug_shift = model_accuracy(augmented, xv_shift, yv)
    required = [clean_clean, clean_shift, aug_clean, aug_shift]
    eligible = all(math.isfinite(float(x)) for x in required)

    decision_row = {
        "seed": int(seed),
        "eligible": bool(eligible),
        "clean_teacher_clean_validation_accuracy": clean_clean,
        "clean_teacher_shifted_validation_accuracy": clean_shift,
        "augmented_teacher_clean_validation_accuracy": aug_clean,
        "augmented_teacher_shifted_validation_accuracy": aug_shift,
    }

    return {
        "experiment": "C68E_TEACHER_VALIDITY_REPAIR_EXPLORATION",
        "evidence_class": "PROSPECTIVE_EXPLORATORY",
        "status": "FRESH_SEED_OUTCOME" if seed in FRESH_SEEDS else "IMPLEMENTATION_VERIFICATION_OUTCOME",
        "seed": int(seed),
        "eligible": bool(eligible),
        "test_evaluated": False,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "protocol": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "target_sigma": TARGET_SIGMA,
            "augmentation_fraction": AUGMENT_FRACTION,
            "permutation_seed": seed + PERM_OFFSET,
            "augmentation_seed": seed + AUGMENT_OFFSET,
            "validation_epsilon_seed": seed + VAL_EPS_OFFSET,
        },
        "training_provenance": training_prov,
        "provenance_hashes": {
            "validation_standard_normal_sha256": sha256_tensor(val_eps),
            "shifted_validation_tensor_sha256": sha256_tensor(xv_shift),
            "clean_teacher_final_state_sha256": state_dict_sha256(clean),
            "augmented_teacher_final_state_sha256": state_dict_sha256(augmented),
        },
        "metrics": decision_row,
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
