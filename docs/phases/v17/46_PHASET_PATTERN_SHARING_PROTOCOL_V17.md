# Phase T — Pattern-index sharing below 76 bytes (v17)

## Question
Can the confirmed 76 B 1:4 / 2-bit representation be reduced further by sharing the 1-of-4 support index across output channels while keeping the same number of stored 2-bit coefficients?

## Independent cohort
First 8 baseline-eligible seeds (clean >=0.95) ascending from seed 2600; independent of all prior low-bit cohorts.

## Fixed representation
Each output still stores 18 selected kernel coefficients plus one bias (152 stored scalar values total) at signed 2-bit. A single calibrated FP16 scale is shared globally. Coefficients on the fixed support are ridge-refit.

Support-index sharing across output channels:
- group size 1: separate pattern per output; 288 pattern bits; 76 B reference
- group size 2: pattern shared by output pairs; 144 pattern bits; 58 B
- group size 4: pattern shared by groups of four outputs; 72 pattern bits; 49 B
- group size 8: one pattern shared across all outputs; 36 pattern bits; 44.5 B

Within each output group and each 1:4 group, the shared index is chosen from the FP32 fitted Conv3 by summed squared weight magnitude across outputs in that output group. No task-accuracy outcome is used for support selection.

## Repair
Exactly tau=8 full-shell repair, frozen compiled core, matched tau=8 continued-training control.

## Primary endpoints
Mean matched-control utility, seed-bootstrap 95% CI, PASS95 fraction. A representation is stable only if lower 95% CI >=0.95.