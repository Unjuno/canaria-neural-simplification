# C67E result — extended Gaussian shift severity frontier

## Status

**PROSPECTIVE EXPLORATORY**. Confirmatory claim is not allowed from C67E.

Protocol was locked before outcomes at commit `81c353bf576c4d6eb8b9a75e815c78b33a4c2199`. Fresh model seeds were `65400–65415`; all 16 were eligible. Held-out test data were not used.

## Terminal decision

`STOP_VALIDITY_BOUNDARY_AT_SIGMA_0_36`

The locked P0-versus-P2 gates passed at every tested sigma (`0.20, 0.28, 0.36, 0.44, 0.52, 0.60`), but the prospectively locked teacher-task validity safeguard first failed at sigma `0.36`. Therefore C67E did **not** select a P0 frontier sigma for confirmation.

| sigma | P0-P2 validation mean (pp) | validation 95% CI (pp) | P0/P2 NMSE geomean | NMSE 95% CI | teacher drop mean (pp) | teacher-drop 95% CI (pp) | validity |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0.20 | -0.462963 | [-1.134259, 0.162037] | 1.028127 | [1.012060, 1.043685] | -6.388890 | [-6.805556, -5.949076] | PASS |
| 0.28 | -0.162037 | [-0.648148, 0.347222] | 1.021286 | [1.008823, 1.033976] | -15.763890 | [-16.967594, -14.606482] | PASS |
| 0.36 | +0.208332 | [-0.833335, 1.365740] | 1.026591 | [1.015040, 1.038910] | -27.638890 | [-29.513891, -25.833334] | FAIL |
| 0.44 | +0.092593 | [-0.972221, 1.250001] | 1.048498 | [1.031709, 1.067713] | -40.231482 | [-42.199075, -38.379630] | FAIL |
| 0.52 | +0.092592 | [-1.018519, 1.249999] | 1.068269 | [1.048189, 1.088050] | -50.347222 | [-52.152778, -48.472223] | FAIL |
| 0.60 | +0.972223 | [-0.046295, 2.037037] | 1.089712 | [1.061469, 1.115576] | -57.638890 | [-59.328705, -55.972224] | FAIL |

P0 validation non-inferiority used the locked `-2 pp` margin; P0/P2 NMSE used the locked `1.25` ratio margin. The teacher-task validity safeguard required the paired shifted-minus-clean teacher accuracy bootstrap lower bound to remain above `-20 pp`. P2 reference validity used a `-5 pp` lower-bound margin and remained PASS across the grid.

## Scientific interpretation

The clean teacher averaged `97.96%` validation accuracy. At sigma `0.36`, shifted-teacher accuracy averaged about `70.32%`, a mean drop of `27.64 pp`; its bootstrap interval was entirely beyond the locked `-20 pp` validity threshold. P2 remained within the separate reference-validity margin at that sigma (mean P2-minus-teacher gap about `-3.29 pp`, bootstrap95 about `[-4.17,-2.38] pp`).

Thus the first blocker was **teacher task degradation**, not evidence that P0 lost non-inferiority to P2. Although P0 continued to pass both numerical P0-versus-P2 gates at sigma `0.36` and beyond, those higher-severity comparisons are not used to select a frontier because the target teacher is already outside the preregistered task-valid regime.

## Safe statement

> In the repository Residual-MLP testbed, C67E found that the preregistered teacher-task validity boundary was crossed at Gaussian sigma `0.36` before any interpretable P0-versus-P2 non-inferiority failure was observed. P0 remained jointly non-inferior to P2 numerically across the tested grid, but points at and above the validity boundary are descriptive only for the interface-frontier question.

## Not supported

- P0 is confirmed sufficient through sigma `0.60`;
- sigma `0.36` is a P0 failure threshold;
- teacher correction is universally unnecessary under strong distribution shift;
- the `-20 pp` teacher validity threshold is a universal scientific constant;
- the true continuous teacher robustness boundary is exactly sigma `0.36`;
- conclusions about the imported Residual CNN C59/C60 line.

## Next valid experiment

Do **not** continue increasing sigma while keeping the same degraded clean-trained teacher as the target. The next experiment should redesign the reference question: determine whether a shift-adapted/shift-trained teacher restores task validity at high sigma, and only then re-evaluate interface complexity against that valid target. This must be a separately named prospective experiment, not C68R confirmation, because C67E selected no P0 frontier candidate.
