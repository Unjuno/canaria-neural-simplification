# Phase H — precision × stored-weight-count protocol (v14)

Frozen before Phase-H outcomes.

## Goal
Measure the replacement Conv3 function under two independent controls:
1. bits per retained scalar;
2. number K of retained scalars.
Then measure their joint effect and whether 2-epoch full-shell repair lowers the required K.

## Architecture / data
Same residual 8-block sklearn-digits model used in v11–v13: ch=8, residual scale=0.5, base epochs=24, AdamW lr=3e-3, weight decay=1e-4. Baseline eligibility clean accuracy >=0.95.

## Seeds
Scan upward from seed 1700 and take the first 8 eligible seeds. Eligibility is determined from baseline clean accuracy only, before any replacement outcome is computed.

## Reference replacement
Fit one linear Conv3 (8->8, bias; 584 FP32 scalars) from full-core input to full-core output using the first 192 training examples and ridge=1e-5. Candidate remains frozen.

## H1: fine precision sweep
All 584 scalars retained. Symmetric uniform signed quantization, one FP32 scale per candidate, bits b in {2,3,4,5,6,8,10,12,16,32}. This is a fixed-grid numerical experiment, not a claim about hardware FP formats.

## H2: joint K × bit sweep, tau=0
Retain the K largest-magnitude scalars of the fitted Conv3 (including bias scalars), with K in {64,96,128,160,192,224,256,288,320,352,384,448,512,584}. Quantize the retained values with b in {2,3,4,6,8,12,16,32}. Missing values are exactly zero. No shell repair.

## H3: repair-rescue subset
For K in {64,128,192,256} and b in {4,8,12,32}, freeze the sparse/quantized Conv3 and permit the full shell (stem, b_in, b_out, head) to repair for tau=2 epochs. Compare against an uncompiled matched control receiving exactly the same augmented 2-epoch continuation. Utility is candidate augmented accuracy / matched-control augmented accuracy.

## Metrics
- baseline clean / augmented accuracy
- reference span relative MSE
- no-repair augmented utility = candidate_aug / baseline_aug
- repaired augmented utility = candidate_aug / matched_control_aug
- PASS95 = utility >=0.95
- nominal per-function code bits = retained-value bits + index bits + 32-bit scale; index uses ceil(log2(584))=10 bits per retained scalar. For FP32 sparse values no scale is charged.

## Primary guardrails
- Do not mix bit precision with K scalar count.
- No post-hoc seed replacement based on replacement results.
- No repair is used in H1/H2.
- H3 is explicitly adaptive and must not be interpreted as intrinsic simplification.