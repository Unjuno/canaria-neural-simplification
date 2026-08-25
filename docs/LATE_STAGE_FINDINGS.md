# Late-stage findings: G18–G27

This document summarizes the mechanism/controller experiments run after the G7–G17 training-time consolidation mainline.

These results use the same small real-text character-LM testbed unless stated otherwise. They should not be read as evidence for large pretrained language models.

## G18 — deadline-aware autonomous consolidation — PASS

Fresh seeds: 6200–6211, `n=12`.

Compared with the static NMSE controller:

- mean test PPL: `19.9384 → 19.7574`;
- paired mean difference: `−0.1810`;
- 95% paired bootstrap CI: `[-0.3508, -0.0417]`;
- mean compiler updates: `184 → 136`;
- final 2-block target reached: `12/12`.

Interpretation: remaining learning horizon contains useful information that is missing from an instantaneous static fidelity threshold.

Important boundary: earlier commit did not reduce compiler cost in every seed; some early commits made the later contraction harder.

## G19 — staged-path generalization on 5-block family — PASS

Fresh seeds: 6600–6607, `n=8`.

Comparison:

- staged `5→4→2`;
- direct `5→2`;
- identical total compiler updates: `192` per condition.

Result:

- mean staged PPL: `19.3580`;
- mean direct PPL: `20.0475`;
- staged − direct: `−0.68945`;
- 95% CI: `[-0.82515, -0.58039]`;
- staged wins: `8/8`.

Interpretation: the staged/recontracting advantage is not unique to the original `4→3→2` path.

## G20d — next-compiler optimization cost after recontracting — PASS

Fresh seeds: 7300–7307, `n=8`.

At fixed standardized MSE target `0.03` for the next `3→2` compiler:

- PRE mean updates: `41.5`;
- POST-recontract mean updates: `32.125`;
- difference: `−9.375`;
- 95% CI: `[-12.5, -6.75]`;
- POST fewer: `8/8`;
- mean relative reduction: about `22%`.

Interpretation: after intermediate-model task learning, the next compiler reaches the same scale-controlled functional-error target with fewer optimizer updates.

## G20e — matched normalized error is not equally task-safe — confirmed boundary

Fresh seeds: 7500–7507, `n=8`.

At the same standardized-error threshold:

- PRE mean immediate NLL damage: `0.00718`;
- POST mean immediate NLL damage: `0.01062`;
- difference: `+0.00344`;
- 95% CI: `[+0.00179, +0.00514]`.

At the same time POST required fewer fit updates.

Interpretation: recontracting makes the compiler easier to fit in normalized function space, but the matured task computation becomes more sensitive to residual approximation error.

## G21 — hard shadow-damage veto — FAIL

Fresh seeds: 7800–7811, `n=12`.

A hard task-damage veto was added to the deadline-aware policy.

Result:

- final 2-block target reached in only `10/12`;
- mean compiler updates increased from `120` to `136`;
- two runs ended at depth 3.

The all-run PPL comparison is not capacity-matched and must not be treated as a valid utility win.

Interpretation: "do not commit when shadow damage is high" is too conservative as a hard rule. Refusing contraction also has cost.

## G22 — sensitivity increase after recontracting — PASS

Fresh seeds: 8100–8111, `n=12`.

POST − PRE:

- downstream gradient RMS: `+0.01736`, 95% CI `[+0.01597, +0.01886]`, 12/12 positive;
- logit amplification per standardized error: `+0.18835`, CI `[+0.17563, +0.20135]`, 12/12 positive;
- matched-error NLL damage: `+0.00227`, CI `[+0.00091, +0.00372]`.

Compiler fit updates decreased from `42.25` to `32.33` on average.

Interpretation: the "fit easier / task more sensitive" dual effect is reproducible.

## G23 — error direction predicts task damage better than error magnitude — PASS

Fresh seeds: 8300–8311, 192 observations.

Leave-one-seed-out prediction:

- standardized error only MAE: `0.004303`;
- signed first-order `gradient · error` MAE: `0.003189`;
- seed-mean MAE difference: `−0.001114`;
- 95% CI: `[-0.001701, -0.000538]`;
- candidate better: `10/12` seeds.

Interpretation: residual-error direction relative to task sensitivity matters, not only its norm.

## G24 — first + second-order risk proxy — PASS

Fresh seeds: 8400–8411, 192 observations.

Leave-one-seed-out MAE:

- first-order signed term only: `0.002763`;
- first-order + squared logit-error term: `0.000548`;
- difference: `−0.002216`;
- 95% CI: `[-0.002669, -0.001805]`;
- improved: `12/12`.

The residual after the first-order term had Spearman `ρ≈0.978` with squared logit error.

Descriptive all-data coefficients were approximately:

`ΔNLL ≈ -7.6e-6 + 1.029 * first_order + 0.770 * logit_error²`

These coefficients are empirical, not a mathematical theorem.

## G25 — fixed risk model transfers to 5-block family — PASS

Fresh seeds: 8500–8507, `n=8`.

The G24 coefficients were frozen and transferred without refitting.

Mean seed MAE:

- signed-only predictor: `0.010334`;
- first + second-order predictor: `0.001999`;
- difference: `−0.008335`;
- 95% CI: `[-0.009464, -0.007210]`;
- improvement: `8/8`.

Interpretation: the risk decomposition is not merely an in-sample fit to the original depth path, although broader architectural validity remains open.

## G26 — horizon-aware future-damage prediction — PASS

Fresh seeds: 8700–8711, `n=12`.

The recovery/horizon coefficients were frozen before confirmatory evaluation.

Seed-level MAE:

- immediate-risk reused as future-damage baseline: `0.002693`;
- horizon-corrected predictor: `0.001159`;
- difference: `−0.001534`;
- 95% CI: `[-0.001887, -0.001149]`;
- improvement: `12/12`.

At four task epochs ahead, MAE decreased from about `0.00498` to `0.00183`.

Important mechanistic result: damage does not simply decay monotonically. Relative damage depends on candidate risk, matched-teacher improvement, and horizon.

## G27 — fixed risk-cap budget selection — exploratory, no Pareto result

A compiler-budget selector used the G26 risk predictor at a fixed future-damage cap.

- strict cap used more compiler updates and slightly improved PPL;
- loose cap used fewer updates and slightly worsened PPL.

Conclusion: prediction quality does not automatically solve the cost/utility optimization problem. A fixed scalar risk cap is not enough.

No confirmatory G27 claim is made.

## Combined interpretation

The late-stage evidence supports a two-sided view of recontracting:

1. **optimization side:** a matured intermediate model can be easier to compile to a given normalized functional fidelity;
2. **task side:** the downstream computation can become more sensitive to the remaining approximation error.

A useful future controller therefore needs to consider:

- compiler residual magnitude;
- residual direction relative to task gradient;
- logit-space nonlinear amplification;
- remaining learning horizon;
- compiler cost;
- the opportunity cost of refusing contraction.

This mechanism work refines, rather than replaces, the core compositional-simplification claim.
