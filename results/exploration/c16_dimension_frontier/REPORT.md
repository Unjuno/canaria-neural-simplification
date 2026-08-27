# C16 teacher-correction dimension frontier — exploratory report

C16 varies the self-anchored teacher correction dimension while holding the recursive hierarchy, optimization, and 4096-parameter budgets fixed.

Fresh seeds: `1470–1472`. Three nested basis families per seed: identity plus random orthogonal bases `20261001` and `20261002`. No held-out test evaluation.

## Worst-basis fidelity by teacher dimension

| correction dimension | worst/full-64 ratio across seeds | all 9 basis×seed conditions improve frozen? |
|---:|---:|---|
| 8 / 64 | 1.385–1.476x | yes |
| 16 / 64 | 1.312–1.410x | yes |
| 24 / 64 | 1.251–1.310x | yes |
| 32 / 64 | 1.196–1.223x | yes |

All **36/36** self-anchored conditions improved over the frozen hierarchy. Fidelity improves smoothly with teacher correction dimension.

The notable exploratory result is that even an 8-dimensional correction carries a repeatable repair signal, but its functional penalty relative to full hidden alignment is much larger. A 16-dimensional correction is a useful next confirmatory target because it represents only one quarter of the hidden interface while remaining in the roughly 1.31–1.41x worst-basis range in this exploration.

C16 does not establish a universal minimum sufficient dimension.
