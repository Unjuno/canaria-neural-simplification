# Phase S — Independent confirmation of 76-byte representation (v17)

## Primary hypothesis
The 1:4 semi-structured, 2-bit Conv3 representation with ONE FP16 shared scale (76 B nominal total) can maintain matched-control utility after tau=8 shell repair.

## Cohort
First 8 baseline-eligible seeds (clean >= 0.95) in ascending order from seed 2500. Fully independent of Phase P/Q/R.

## Fixed conditions
Same support/refit/quantization procedure as Phase P/Q.
- 1 shared FP16 scale = 76 B PRIMARY
- 2 shared scales = 78 B secondary
- 8 per-output scales = 90 B reference

## Repair
Exactly tau=8 full-shell repair, core frozen, against tau=8 matched continued-training control. No tau search in this cohort.

## Primary PASS
76 B is confirmed stable iff seed-bootstrap lower 95% CI of mean matched-control repair utility >= 0.95. PASS95 fraction is secondary. No conditions change after hash lock.