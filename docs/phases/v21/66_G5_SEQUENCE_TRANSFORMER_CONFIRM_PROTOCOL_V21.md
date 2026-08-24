# G5 Sequence Transformer Encoder Confirmatory Protocol — v21

## Lock point
This protocol is written after pilot seed 3299 and before any seed >=3300 outcome is inspected.

## Fixed condition
No further architecture/candidate/repair search is permitted on the confirmatory cohort.
- fixed synthetic sequence-order dataset from v21 pilot
- teacher: 4 Transformer encoder blocks, d=24, heads=4, MLP=48
- compiler: 2 Transformer blocks, d=24, heads=4, MLP=24
- 512 unlabeled calibration examples
- 60 epochs residual-stream MSE fit
- compiler core frozen during task repair
- shell-only repair: token embedding, CLS, positional embedding, final norm, task head
- tau={0,2,8}
- matched continued-training controls for tau>0

## Eligibility and seed queue
- seeds begin at 3300 and increase monotonically
- baseline eligibility is held-out accuracy >=0.95
- use the first 8 eligible seeds only
- eligibility is based on baseline performance only, before simplification outcome inclusion

## Primary zero-shot hypothesis
Let U0 = compiled tau=0 accuracy / baseline accuracy for each eligible seed.

**Z PASS** iff the seed-bootstrap 95% CI lower bound of mean U0 is >=0.95.

If Z fails, zero-shot sequence transfer is rejected for this condition.

## Adapted-transfer fallback
If Z fails, evaluate tau=8 matched-control utility.

**A PASS** iff the seed-bootstrap 95% CI lower bound of mean U8 is >=0.95.

Tau=2 is prespecified secondary evidence and is reported regardless of Z/A outcome.

## Whole-network accounting follow-up
Only after Z or A passes, an explicit q8 real-byte state-stream codec may be evaluated on the same fixed 8 seeds. The architecture/decoder program is shared and not charged per model; headers, tensor names/shapes, scales, and parameter payload are charged.

## Statistical unit
Training/model seed. Bootstrap resamples seeds, not per-example predictions.

## Interpretation
- Z PASS: frozen G3 Transformer compiler transfers zero-shot from vision tokens to non-image sequence tokens.
- Z FAIL + A PASS: sequence setting requires bounded task repair but no new compiler search.
- both fail: N under this frozen transfer budget; do not retune the same cohort.
