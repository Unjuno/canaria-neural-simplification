# C73E: full-pipeline calibration repair

Evidence: PROSPECTIVE_EXPLORATORY. Main/public claims unchanged.

Decision: `COMPILER_REPAIR_NOT_ESTABLISHED`.

All planned seeds must be present; PASS/FAIL/UNCERTAIN are distinct. A failed non-inferiority gate is not proof of inferiority.

| Cell | Mean gap to robust teacher (pp) | 95% CI (pp) | Gate |
|---|---:|---|---|
| N192_D64 | -5.069445 | [-6.041666120290756, -4.120369628071785] | UNCERTAIN |
| N192_H64 | -6.504630 | [-7.569443807005882, -5.532407388091087] | FAIL |
| N192_S64 | -5.462964 | [-6.59722276031971, -4.3055567890405655] | UNCERTAIN |
| N384_D64 | -2.939815 | [-4.050926119089127, -1.8055543303489685] | PASS |
| N384_H64 | -3.981482 | [-4.95370551943779, -3.0787046998739243] | PASS |
| N384_S64 | -3.958334 | [-5.162038654088974, -2.847222238779068] | UNCERTAIN |

## Descriptive paired contrasts

- N192_D64_minus_S64: 0.393519 pp, 95% [-0.1388896256685257, 0.9259264916181564]
- N192_S64_minus_H64: 1.041666 pp, 95% [0.138888880610466, 1.9212953746318817]
- N384_D64_minus_S64: 1.018519 pp, 95% [0.4861094057559967, 1.5740737318992615]
- N384_S64_minus_H64: 0.023148 pp, 95% [-0.7175933569669724, 0.7638897746801376]
- N384_minus_N192_D64: 2.129629 pp, 95% [1.5046298503875732, 2.77777798473835]
- N384_minus_N192_H64: 2.523147 pp, 95% [1.64351686835289, 3.3333327621221542]
- N384_minus_N192_S64: 1.504629 pp, 95% [0.5324076861143112, 2.5462958961725235]

## Boundaries
No reduced dimension selected. This is Residual-MLP only. Seed intervals are conditional on the fixed digits split and fixed nested calibration sets. Raw MSE and separate denominators are recorded; independently normalized NMSE values are not a common-scale generalization gap. Each mapping fit uses 600 updates and batch128, hence 76800 draws at both sample sizes.

Preflight reproduces C72E direct-path teacher/data/state hashes and metrics exactly on bridge70300 in the same pinned CPU environment. No held-out test was used.
