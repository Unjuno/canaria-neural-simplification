# Phase Z — Whole-network head compression protocol (v19)

Status: preregistered before outcome evaluation.

## Goal
Test whether the v18 4-bit compiled whole network (~13.25 KB) can be reduced below 10 KB by compressing the classifier head while preserving function.

## Cohort
Starting at seed 2900, use the first 8 seeds whose baseline clean accuracy is >= 0.95. No replacement/compression outcome is inspected for ineligible seeds.

## Base/compile procedure
Identical to v18 Phase X/Y: residual 8-block digits model, 24 baseline epochs, full-span shared-pattern 1:4 ternary 44.5 B core, full-shell repair tau=8, matched continued-training control tau=8.

## Head compression
After shell repair, factorize the first head matrix W in R^(48x512) by exact SVD truncation. Implement W_r = B_r A_r with no nonlinearity between A_r and B_r, keep the original bias on B_r, then original ReLU and final 48->10 linear layer.
Ranks: 16, 24, 28, 32, 36.

All non-core stored tensors are then quantized with the same per-tensor calibrated signed 4-bit quantizer used in v18 Phase Y. Core stays as the exact 356-bit (44.5 B) code.

## Primary condition
rank=32, 4-bit, no additional repair. This rank is chosen before outcomes because nominal whole-model storage is < 10,000 bytes.

## Primary endpoints
1. Combined fidelity = accuracy(lowrank-4bit) / accuracy(compiled-FP32) on the fixed augmented validation set.
2. Quantized utility = accuracy(lowrank-4bit) / accuracy(matched-control-4bit).
3. Nominal packed bytes including 4-bit weights, FP16 per-tensor scales, 356-bit core, and 2 bytes low-rank architecture metadata.

## Decision
PASS if for rank32:
- mean packed bytes < 10,000;
- seed-cluster bootstrap 95% CI lower bound of combined fidelity >= 0.95;
- seed-cluster bootstrap 95% CI lower bound of quantized utility >= 0.95.
Otherwise FAIL/UNCERTAIN according to which criteria fail.

## Secondary/exploratory
Ranks 16/24/28/36 map the storage-fidelity curve. Additional head-only repair is not part of the primary test and may only be run after the primary decision.