# C64R — P0/P1/P2 task-weighted frontier exploration

Evidence class: **PROSPECTIVE_EXPLORATORY**

Decision: **`ADVANCE_P0_TO_C65R`**

## Locked primary frontier

Fresh seeds: `62400–62415`; eligible `16/16`; held-out test unused.

### P0 vs P2

- shifted-validation accuracy, P0−P2 mean: **−0.115741 pp**
- paired bootstrap95: **[−0.578704, +0.393518] pp**
- exploratory non-inferiority margin: `−2 pp` → PASS
- P0/P2 NMSE geometric mean: **1.010043**
- paired bootstrap95: **[0.994568, 1.026083]**
- exploratory ratio margin: `1.25` → PASS
- joint gate: PASS

The prospectively locked decision tree therefore selects P0 for a separate fresh C65R confirmation.

### P1 vs P2

P1 also passed both exploratory gates:

- validation P1−P2 mean: **+0.092593 pp**
- bootstrap95: **[−0.277777, +0.462963] pp**
- P1/P2 NMSE geometric mean: **1.006484**
- bootstrap95: **[0.999378, 1.013818]**

P0 was selected because the decision tree tests the smaller candidate first.

## What P0 means

P0 is **not** a trained zero-dimensional correction module. It is the frozen base recursive hierarchy compiled directly, with no teacher-residual correction and no top-boundary adaptation. This avoids an AdamW-weight-decay confound that would move parameters even under a zero correction target.

## Task-weighted diagnostics

On shifted validation, mean residual-geometry diagnostics were approximately:

- P1 Euclidean capture: `0.1335`
- P2 Euclidean capture: `0.2387`
- P1 logit-L2 retained ratio: `0.8391`
- P2 logit-L2 retained ratio: `0.7057`
- P1 local-Fisher retained ratio: `0.8878`
- P2 local-Fisher retained ratio: `0.7619`

The candidate corrections therefore remove measurable activation/logit/Fisher-weighted residual components even though P0 remains within the exploratory utility/NMSE non-inferiority margins versus P2. Spearman associations between these diagnostics and downstream P0/P1-vs-P2 gaps were weak to modest in this 16-seed cohort and are mechanism-generating only.

A Fisher retained ratio may exceed 1 because the QR correction is Euclidean rather than Fisher-orthogonal; cross-terms in the Fisher quadratic can increase after subtracting the projected component. It must not be interpreted as a bounded explained-energy fraction.

## Interpretation boundary

C64R does **not** confirm that P0 is sufficient. It only selects P0 for a separate prospective fresh confirmation. Even a future C65R PASS would remain scoped to the exact Residual-MLP recursive-hierarchy / Gaussian `sigma=.04` / 192-calibration / compilation protocol and would not establish that teacher correction is universally unnecessary.

This evidence line is separate from the imported Residual CNN C59/C60 experiments.
