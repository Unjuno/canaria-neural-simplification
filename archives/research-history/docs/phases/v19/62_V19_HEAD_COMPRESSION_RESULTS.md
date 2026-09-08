# v19 Head Compression Results

## Goal
Reduce the v18 whole-network 4-bit model (~13.25 KB) below 10 KB without losing confirmatory utility.

## Phase Z — rank32 per-tensor 4-bit, 9,921.5 B
Primary test failed. Mean combined fidelity ~0.941 (95% CI lower ~0.927) and mean quantized utility ~0.943 (lower ~0.919). Rank truncation itself was high (~0.992), so instability was mainly quantization of the factorized head.

## Z2 — two extra head-only epochs
Failed to stabilize quantization. Mean quantized utility ~0.924, lower CI ~0.886.

## Z3/Z4 — rank31, channelwise 4-bit, FP16 biases, 9,968.5 B
Exploratory cohort improved substantially. Independent Phase-Z4 cohort showed:
- combined fidelity mean 0.9537, 95% CI [0.9405, 0.9675] — FAIL full criterion;
- compression fidelity vs dense compiled q4 mean 0.9658, [0.9524, 0.9787] — PASS head-compression-specific criterion;
- quantized utility mean 0.9540, [0.9199, 0.9807] — FAIL.

Rank sweep to rank36 (~11.38 KB) did not fully solve whole utility, although compression relative to dense q4 remained stable.

## Phase AA — 2:4 head
Calibration refit was ill-conditioned and failed after 4-bit quantization. Removing refit stabilized the head: 9,674.5 B, compression fidelity vs dense q4 lower CI ~0.955, but whole-model combined fidelity/utility still failed.

## Phase AB — less aggressive 296 B Conv3-q4 core + 2:4 head
Exploratory cohort at 9,926 B:
- combined fidelity mean 0.9629, 95% CI [0.9501, 0.9784] — PASS;
- compression fidelity vs dense q4 0.9768 [0.9615, 0.9923] — PASS;
- whole quantized utility 0.9627 [0.9459, 0.9783] — narrowly below threshold.

## Phase AB independent confirmation — seeds 3100–3107
Frozen 9,926 B condition independently confirmed:
- combined fidelity mean 0.9644, 95% CI [0.9514, 0.9754];
- compression fidelity vs dense q4 mean 0.9742, [0.9644, 0.9840];
- matched-control quantized utility mean 0.9842, [0.9640, 1.0069].
All preregistered thresholds pass.

## Phase AD — exact 9,926-byte codec
Implemented exact binary serialization:
- 296 B dense Conv3 q4 core;
- channelwise q4 stem/b_in/b_out and final classifier layer, FP16 biases/scales;
- first classifier layer 2:4 sparse, q4 retained values, exact 3-bit 2-of-4 pattern code;
- 2-byte metadata header.

All eight confirmatory models serialize to exactly 9,926 bytes. Decode/re-encode round-trip logit max difference is 0 for all seeds.
Using the decoded binary models directly:
- combined fidelity = 0.96363, 95% CI [0.95160, 0.97401];
- vs dense compiled q4 = 0.97350, [0.96412, 0.98328];
- matched-control quantized utility = 0.98347, [0.96393, 1.00589].

## Interpretation
The <10 KB boundary is achievable, but not by uniformly pushing every component to its smallest representation. The successful design backs off the core from 44.5 B to a still-small 296 B Conv3 q4 operator while using structured 2:4 sparsity in the classifier head. This supports a component-allocation view: minimum total code is reached by distributing representational budget unevenly across core and task-aligned head.
