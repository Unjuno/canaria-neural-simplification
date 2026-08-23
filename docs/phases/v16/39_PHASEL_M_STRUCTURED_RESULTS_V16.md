# Phase L/M results — structured low-bit weights (v16)

## Phase L exploratory, 8 eligible seeds
Key no-repair results:
- dense 3-bit: 235 B, mean retention 0.9868.
- kernel-block R=32, 3-bit: 135 B, retention 0.9957.
- 2:4 semi-structured, 3-bit: 181 B, retention 0.9992.
- spatial-offset P=5, 4-bit: 181.125 B, retention 1.0014.

Pareto conditions were locked before repair (SHA256 recorded in raw_experiments/phaseL_structured_v16/repair_selection_locked.json.sha256).
Tau=2 matched-control repair:
- kernel-block R=24, 3-bit, 108 B: utility 0.9662, 95% bootstrap CI [0.9501, 0.9845], PASS95 7/8.
- kernel-block R=32, 3-bit, 135 B: utility 0.9708, CI [0.9509, 0.9885], PASS95 6/8.
- spatial-offset P=6, 3-bit, 164.125 B: utility 0.9938, CI [0.9818, 1.0090], PASS95 8/8.
- 2:4 semi-structured, 3-bit, 181 B: utility 0.9864, CI [0.9745, 0.9991], PASS95 8/8.

## Phase M independent holdout, new 8 eligible seeds 2000-2007
No-repair:
- dense 3-bit: 235 B, retention 0.9932, 95% CI [0.9755, 1.0068], utility 0.9698.
- 2:4 semi-structured, 3-bit: 181 B, retention 0.9999, CI [0.9906, 1.0100], utility 0.9766, PASS95 7/8.
- spatial-offset P=5, 4-bit: 181.125 B, retention 0.9830, CI [0.9786, 0.9871], utility 0.9601.
- kernel-block R=28, 3-bit: 121.5 B, retention 0.9808, CI [0.9618, 0.9948], utility 0.9576.
- kernel-block R=24, 3-bit: 108 B, retention 0.9598; the stronger Phase-L no-repair result did not replicate.
- kernel-block R=32, 3-bit: 135 B, retention 0.9743; the stronger Phase-L no-repair result did not replicate.

Tau=2 matched-control repair on the prelocked primary conditions:
- kernel-block R=24, 3-bit, 108 B: utility 0.9709, CI [0.9519, 0.9860], PASS95 7/8.
- kernel-block R=32, 3-bit, 135 B: utility 0.9710, CI [0.9507, 0.9881], PASS95 7/8.
- 2:4 semi-structured, 3-bit, 181 B: utility 0.9920, CI [0.9708, 1.0097], PASS95 7/8.
- spatial-offset P=5, 4-bit, 181.125 B: utility 0.9771, CI [0.9589, 0.9938], PASS95 7/8.

## Storage interpretation
- FP32 Conv3 reference = 2336 B.
- 2:4 3-bit = 181 B = 92.25% smaller than FP32 Conv3 and 22.98% smaller than dense 3-bit.
- kernel-block R=24 3-bit = 108 B = 95.38% smaller than FP32 Conv3 and 54.04% smaller than dense 3-bit.

## Main empirical conclusion
Structured support matters. 2:4 semi-structured 3-bit is the strongest no-repair low-storage point validated on an independent cohort. The 108 B kernel-block model is not stable without repair, but its tau=2 matched-control recovery replicated in two independent 8-seed cohorts.
