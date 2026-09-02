# C65R — P0 versus P2 prospective confirmation

Evidence class: **PROSPECTIVE_CONFIRMATORY**

Decision: **`C65R_CONFIRMATORY_PASS`**

## Locked design

The protocol was locked before fresh outcomes at commit `2196a84c65e3183a65b41cc0f8b48e6862772192`.

- fresh seeds: `63400–63415`
- verification seed: `63300` (preflight PASS before fresh launch)
- Residual-MLP recursive hierarchy
- additive Gaussian input shift `sigma=.04`
- fixed 192-sample calibration subset
- P0 versus P2 only; no P1 or other rescue condition
- validation non-inferiority margin: `−2 pp`
- P0/P2 NMSE geometric-mean ratio margin: `1.25`
- paired percentile bootstrap: `100000`, RNG `2237321090`
- held-out test unused

P0 means the frozen recursive hierarchy was compiled directly, with **no teacher-residual correction and no top-boundary adaptation**. P2 uses the first two columns of the canonical nested QR basis of the shifted-calibration teacher residual, followed by the locked top-boundary adaptation.

## Confirmatory result

- attempted / eligible: **16 / 16**
- missing rows: **0**
- validation P0−P2 mean: **−0.370370 pp**
- bootstrap95: **[−0.578703, −0.162037] pp**
- preregistered validation margin: `−2 pp` → **PASS**
- P0/P2 NMSE geometric mean: **0.987108**
- bootstrap95: **[0.973760, 0.999360]**
- preregistered NMSE ratio margin: `1.25` → **PASS**
- held-out test: **not used**

The validation interval is entirely negative: P0 had slightly lower shifted-validation accuracy than P2 in this fresh cohort. The result is therefore a non-inferiority finding, not equality or superiority of P0 on accuracy.

Conversely, the NMSE ratio interval is entirely below 1, so under this cohort and metric the directly compiled P0 condition had lower final NMSE than P2 on average. This does not imply P0 dominates P2 generally; the two outcomes measure different aspects of the bounded task.

## Scientific interpretation

The safe claim is:

> Under the exact C65R Residual-MLP recursive-hierarchy / Gaussian `sigma=.04` / 192-calibration protocol, the directly compiled frozen hierarchy with no teacher-residual correction and no top-boundary adaptation (P0) was non-inferior to the C63R-confirmed P2 condition under the preregistered validation and NMSE margins.

This is stronger than the exploratory C64R signal because C65R used a separate fresh cohort and a prospectively locked P0-versus-P2-only design.

It does **not** establish that teacher correction is universally unnecessary, that zero is a universal minimum interface, that the teacher residual is zero, or that internal representations are equivalent. The result is restricted to this architecture, shift magnitude, calibration regime, replacement/compilation procedure, and decision margins.

This C61R–C65R evidence line is Residual-MLP work and must not be used as confirmation of the imported Residual CNN C59/C60 experiments.
