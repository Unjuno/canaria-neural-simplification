# Phase N — Sub-100-byte structured low-bit test (v17)

## Locked question
Can a full 8-block residual span be replaced by a structured Conv3 representation below ~100 bytes while retaining utility, especially after short shell repair?

## Cohort
Use the first 8 baseline-eligible seeds (clean accuracy >= 0.95) in ascending order starting from seed 2100. Ineligible seeds are retained in metadata and skipped without outcome inspection.

## Baseline / task
Digits, residual 8-block CNN, same training/evaluation pipeline as Phase L/M.

## Fixed candidates
- 2:4 semi-structured, 2-bit (~144 B)
- 1:4 semi-structured, 2-bit (~90 B)
- 1:4 semi-structured, 3-bit (~109 B)
- kernel-block R=16, 3-bit (~81 B)
- kernel-block R=20, 3-bit (~94.5 B)
- kernel-block R=24, 2-bit (~80 B)
- kernel-block R=24, 3-bit (~108 B control)
- spatial-offset P=4, 2-bit (~83 B)
- 2:4 semi-structured, 3-bit (~181 B strong control)

Support is chosen only from the fitted FP32 Conv3 weights using the pre-defined norm/top-N rule. Coefficients on the fixed support are ridge-refit, then per-output-channel calibrated uniform quantization is applied. Per-channel scales are costed as FP16.

## Repair
Matched continued-training control and full-shell repair at tau = 1, 2, 4 epochs. Core remains frozen.

## Primary endpoints
1. no-repair retention relative to same-seed FP32 Conv3;
2. matched-control repair utility;
3. PASS95 fraction;
4. nominal storage including structured pattern/mask metadata and FP16 scale metadata.

## Decision focus
- Stable sub-100 B: mean repair utility >= 0.95 with seed-bootstrap lower 95% CI >= 0.95.
- Promising sub-100 B: mean >= 0.95 but lower CI < 0.95.
- Fail: mean < 0.95.

No candidate, threshold, or tau is modified after this protocol hash is fixed.