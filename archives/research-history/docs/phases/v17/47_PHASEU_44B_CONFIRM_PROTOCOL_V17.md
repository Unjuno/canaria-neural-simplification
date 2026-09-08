# Phase U — Independent confirmation of 44.5-byte representation

## Primary
Confirm the 44.5 B representation discovered under the pre-fixed Phase T family: 1:4 support, 2-bit stored values, one global FP16 scale, and one 1-of-4 pattern per 18 input groups shared across all 8 output channels (36 pattern bits total).

## Cohort
First 8 baseline-eligible seeds (clean >=0.95) in ascending order from 2700. Independent of Phase T.

## Conditions
- 44.5 B PRIMARY (output support pattern shared across all 8 outputs)
- 76 B reference (independent pattern per output)

Both use ridge refit, one actual FP16-rounded calibrated shared scale, frozen core, and exactly tau=8 full-shell repair. Matched tau=8 continued-training control uses identical data order per seed.

## PASS
Primary 44.5 B condition is confirmed stable iff seed-bootstrap lower 95% CI of mean matched-control utility >=0.95.
