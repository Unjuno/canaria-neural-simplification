# Saved-validation threshold robustness: direction persists, selections are fragile

**Post-hoc, descriptive analysis only.** No new data, training, inference or test
metrics. The original accepted criterion and result are unchanged.

## Main finding

Across the **20 of 25 prespecified sensitivity settings with all 16 C/J pairs
observed**, the geometric first-passing replacement-part candidate-budget ratio C/J is below 1,
ranging from **0.676555 to 0.891150**. The direction therefore persists over this
assessed, uncensored threshold surface; it is not confined to the original
NMSE .08 / teacher-relative 2 percentage-point gate.

**The other five settings are censored, not positive results.** With at most
three additional validation errors allowed, teacher 1's C arm has no passing
candidate through budget 6144 under any of the five NMSE ceilings. Its J arm
first passes at 6144. Coverage is 15 both / 0 C-only / 1 J-only / 0 neither in
each case. No full-cohort ratio is computed and no favorable complete-case
subset is substituted.

The original result reproduces: **C/J 0.788324, 11 smaller / 4 tied / 1 larger**.
This is a deterministic robustness description of the same seen cohort, not
independent validation, a new confirmatory success, or a universal threshold
claim. The 20 settings are highly dependent and are not 20 replications.

## Entire fixed sensitivity surface

Each cell is the geometric C/J first-passing candidate-budget ratio. Columns
are maximum additional incorrect classifications relative to the same teacher
among 270 validation examples. The original 2 percentage-point rule is exactly
count-equivalent to the **5-error** column, verified against the saved floating
accuracy rule for all 224 cells. Five errors correspond to 1.851852 percentage
points on this finite sample; the original continuous rule remains 2 points.

| NMSE ceiling | Allow 3 | Allow 4 | Allow 5 | Allow 6 | Allow 7 |
|---:|---:|---:|---:|---:|---:|
| .06 | Censored | .716987 | .676555 | .719326 | .719326 |
| .07 | Censored | .743869 | .771105 | .796740 | .802627 |
| .08 | Censored | .760479 | **.788324** | .794149 | .800016 |
| .09 | Censored | .788324 | .802627 | .808557 | .829309 |
| .10 | Censored | .823226 | .853368 | .875270 | .891150 |

No thresholds are selected as replacements. No new confidence intervals,
p-values, multiplicity-adjusted claims, interpolation or monotonic correction
are used. Every setting's complete 16-seed first-pass list, coverage, per-budget
pass counts and direction counts (where uncensored) are saved in `results/`.

## Selection-margin fragility

At the original first-passing selected budgets, **3/16 C selections and 3/16 J
selections have zero correct-count slack**: one fewer correct prediction, with
the teacher count and all other quantities held fixed, would fail the accuracy
gate. This is a deterministic gate-margin statement, not a resampling result.

At NMSE .08:

| Count gate | C selections changed vs original | J selections changed vs original | C/J geometric ratio | Smaller / tied / larger |
|---|---:|---:|---:|---:|
| Allow 4 (one-error tighter) | 3/16 | 3/16 | .760479 | 12 / 4 / 0 |
| Allow 5 (original-equivalent) | 0/16 | 0/16 | .788324 | 11 / 4 / 1 |
| Allow 6 (one-error looser) | 3/16 | 4/16 | .794149 | 10 / 6 / 0 |

Tightening changes C seeds 1, 12, 13 and J seeds 0, 1, 13. Loosening changes
C seeds 0, 1, 12 and J seeds 0, 1, 5, 12. In particular, teacher 1's C selection
moves 3072→6144 when tightened and 3072→1536 when loosened. The smallest NMSE
slack is **0.000110257** (J, teacher 10, budget 1536); the smallest C NMSE slack
is **0.002559550** (teacher 7, budget 1536). The 32-row margin table reports exact
counts and raw NMSE slack rather than turning these margins into probabilities.

Thus the aggregate direction is stable within the reported uncensored surface,
but individual selected widths and strict-gate coverage are not invariant.

## Method, provenance and tests

- The fixed 25-setting scientific plan is preserved in `PROTOCOL.md` as a
  public extract. Original analysis chronology is recorded in local Git history,
  not an independently timestamped public preregistration. Original hashes and
  the distinction between source and public extracts are documented in
  `SOURCE_PROVENANCE.json` and `TRANSFORMATION.md`.
- Public source: commit `661f1c0648c510ab9303649173dda2bc0997e39e`,
  `research/public_checkpoint_20261027/research/digits_joint_budget_grid/results/evidence/SUMMARY.json`.
  Its scalar projection exactly matches the original accepted input: all 224
  cells and the original minima, geometric ratio and historical interval.
- Portable input is that exact scalar projection plus accepted reference
  metadata; no model arrays are included. `prepare_input.py` verifies the public
  source hash and reproduces it without requiring administrative records.
- All 16 seeds and 224 cells retained. Missing, invalid, duplicate or
  original-gate-inconsistent inputs fail closed.
- Fourteen synthetic tests pass, including inclusive boundaries, censoring,
  nonmonotonic first-pass handling, ordering invariance and an exhaustive
  73,441 teacher/student count-pair check of floating/count gate equivalence.
- A separate implementation independently recomputed all 16 teachers, 224 cells,
  25 settings and 32 margins from saved scalars. All 9,959 structural, numerical
  and output comparisons matched; 16 independent synthetic tests and all 14
  original synthetic tests passed. Check counts are not scientific trial counts.
  This is independent scalar recalculation, not independent model-experiment
  reproduction or validation on unseen examples.

## What is and is not resolved

This narrows a concrete uncertainty: the original descriptive budget advantage
is not isolated to exactly one quality gate within the assessed neighborhood.
It also exposes fragility in selected widths and failure of complete-pair
coverage under the strictest accuracy gate.

It does **not** resolve reused-validation selection bias, uncertainty on new
examples, dataset or teacher generalization, alternative optimizers or
initializations, continuous minimum capacity, architecture/objective causality,
whole-model deployment compression, or hardware speed. Sparse finite budgets,
nonmonotonic passes and one initialization per width/teacher remain. Accuracy
margins cannot replace example-level paired uncertainty analysis, and the
post-hoc surface cannot be used to choose a better-looking primary threshold.

This analysis uses no external data, unseen test evaluation, model loading,
inference or training. It adds no new model experiment or performance benchmark.
