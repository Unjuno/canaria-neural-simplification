# Phase Q — Independent holdout for 76/78-byte scale-shared 1:4 2-bit representation

## Cohort
First 8 baseline-eligible seeds (clean >= 0.95) in ascending order from seed 2400, independent of Phase P.

## Fixed conditions
Same 1:4 support, ridge refit, signed 2-bit values, actual FP16-rounded calibrated scales as Phase P.
- 1 scale group = 76 B primary
- 2 scale groups = 78 B primary
- 8 scale groups = 90 B reference

## Evaluation
No-repair and full-shell tau=2 repair against matched continued-training control.

## Primary criterion
The 76 B condition is independently confirmed stable only if the seed-bootstrap lower 95% CI of tau=2 matched-control utility is >= 0.95. The 78 B condition is secondary confirmatory. No conditions are changed after hash lock.