# Phase G — Replacement-function weight size / precision protocol (v13)

Frozen before outcomes.

## Goal
Separate two questions for the full 8-block replacement function:
1. precision per stored weight coefficient;
2. number of stored per-function coefficients.

## Architecture/data
Same residual 8-block digits setup as v11/v12. Baseline eligibility clean >= 0.95.

## Seeds
Shared dictionary training seeds: 1500..1527 (28).
Held-out evaluation seeds: 1600..1607 (8). No overlap.

## Reference function
Fit a single 3x3 Conv (8->8, bias) to the full core input/output using CALN=192 training examples. The reference has 584 FP32 scalars.

## Experiment G1: precision sweep
Quantize the fitted 584-scalar reference Conv3 with symmetric uniform signed grids at b in {4,8,12,16,32} bits. For b<32 use one per-function FP32 scale; b=32 is unquantized reference. Primary endpoints: validation span relative MSE and no-repair augmented utility. This is a custom fixed-grid b-bit experiment, not a hardware FP4/FP12 datatype claim.

## Experiment G2: coefficient-count sweep
Build a shared PCA dictionary from the 28 dictionary-seed fitted Conv3 vectors. Basis and mean are shared metadata. Held-out functions are represented by K per-function coefficients, K in {0,1,2,4,8,12,16,20,24}. Coefficients are fit in function space on calibration examples with the shared basis frozen. Evaluate FP32 coefficients and fixed shared-grid coefficient quantization b in {4,8,12,16,32}. The shared dictionary storage is reported separately and is not counted as per-function K-float storage.

## Control
A deterministic random orthonormal shared basis of the same dimensionality is evaluated for K in {4,12,24} at FP32.

## Primary interpretation
- G1 isolates coefficient precision at fixed 584 coefficient count.
- G2 isolates per-function coefficient count conditional on an amortized shared dictionary.
- Primary performance uses tau=0 (no shell repair) to avoid shell-capacity confounding.

## Metrics
clean accuracy, augmented accuracy, augmented utility vs original baseline, span relative MSE, PASS95 = utility >= 0.95.

## No post-hoc changes
Seed sets, K grid, bit grid, reference family, calibration size, and PASS threshold are fixed before evaluation outcomes.

## Pre-evaluation eligibility amendment
Before any held-out evaluation seed was run, dictionary seed 1507 failed the pre-existing clean>=0.95 baseline eligibility rule (clean=0.94). Therefore the dictionary construction rule is fixed as: scan integer seeds upward from 1500 and take the first 28 eligible seeds (clean>=0.95). Ineligible seeds are recorded and skipped. Held-out evaluation seeds remain exactly 1600..1607.

## Held-out evaluation eligibility amendment
Before any replacement-function outcome was computed, held-out seed 1600 failed the pre-existing clean>=0.95 eligibility rule (clean=0.94222). The held-out evaluation rule is therefore fixed as: scan integer seeds upward from 1600 and use the first 8 eligible seeds. Eligibility is determined only from original baseline clean accuracy before fitting/evaluating replacement functions.