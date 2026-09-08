# Phase O — Independent sub-100-byte holdout (v17)

## Purpose
Independently validate the Phase N sub-100-byte structured candidates without re-selection.

## Cohort
First 8 baseline-eligible seeds (clean >= 0.95) in ascending order from seed 2200. Seeds are independent of Phase N (2100–2107).

## Locked candidates
- kernel-block 24 x 2-bit = 80 B
- kernel-block 16 x 3-bit = 81 B
- spatial-offset 4 x 2-bit = 83.125 B
- 1:4 semi-structured x 2-bit = 90 B
- kernel-block 20 x 3-bit = 94.5 B
- 2:4 semi-structured x 3-bit = 181 B strong control

Dense 2/3/4/FP32 Conv3 are diagnostic controls.

## Repair
Matched continued-training control; full-shell repair tau = 1 and 2 epochs. Core frozen.

## Primary criterion
For each sub-100-byte candidate at tau=2: mean matched-control utility and seed-bootstrap 95% CI. A condition is called stable only if lower CI >= 0.95. PASS95 fraction is secondary.

No condition may be added or removed after protocol hash lock.