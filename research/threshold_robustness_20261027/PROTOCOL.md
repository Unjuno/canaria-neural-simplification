# Post-hoc validation-threshold robustness — fixed analysis plan

Prepared 2026-10-27 16:59 UTC, before any threshold-sensitivity computation.
This public scientific extract preserves the fixed analysis plan but removes
administrative coordination wording. The plan is post-hoc with respect to
already observed experiment outcomes; it is not an external preregistration
or new experimental evidence.

## Question
Is the observed C/J first-passing candidate-budget advantage dependent on the
particular NMSE .08 / teacher-relative 2 percentage-point quality gate? Reduce
this unresolved scientific issue using recorded validation scalar cells only.
Do not modify the original fixed criterion, selections, or accepted conclusion.

## Fixed input and scope
Input: the 16-seed, 224-cell completed C/J grid at accepted commit
`dbbce006c07e703376dfe7cd0832e5bb3bd2c584`, from
`research/digits_joint_budget_grid/results/evidence/SUMMARY.json`.
A public scalar-equivalent extract is available at commit
`661f1c0648c510ab9303649173dda2bc0997e39e`, under
`research/public_checkpoint_20261027/research/digits_joint_budget_grid/results/evidence/SUMMARY.json`.
Original and public-source hashes are distinguished in `SOURCE_PROVENANCE.json`.
Only JSON scalar `id`, `status`, `teacher_correct`, and each cell's `arm`,
`budget`, `nmse`, `correct`, `n`, `quality_pass`, `reused` are analysis inputs.
All data remain conditional on the same reused digits, validation cohort,
teachers, single width/teacher initialization, optimizer, and finite grid.
No datasets, arrays, weights, model import/inference/training, test450,
unblinding, formal retry, external data, network, paid API, CI, user PC, or
remote modifications.

## Analysis fixed before execution
1. Require all seeds 0–15 and all C/J × budgets
   [512,1024,1536,2048,3072,4096,6144], each exactly once, n=270,
   finite nonnegative NMSE and integer valid counts. Check original quality
   flags against original NMSE≤.08 AND correct/270≥teacher_correct/270−.02.
2. Evaluate exactly 25 settings: NMSE ceilings [.06,.07,.08,.09,.10] crossed
   with maximum additional errors relative to same teacher [3,4,5,6,7].
   In count space pass iff `nmse <= ceiling` AND
   `teacher_correct - correct <= maximum_additional_errors`.
   At n=270 the original 2 percentage-point loss permits exactly 5 additional
   errors; no rounding approximation is needed to express its pass decisions.
3. For every setting show all 16 paired first-passing observed budgets,
   per-budget pass counts, and both/C-only/J-only/neither coverage. No
   interpolation or monotonic corrections. Any no-pass arm is censored at
   the observed candidate grid, not an invented infinite ratio.
4. Only if all 16 pairs are observed report geometric mean C/J and counts of
   smaller/tied/larger C budget. Otherwise suppress the cohort ratio and
   direction count summary rather than use a complete-case rescue.
   No new bootstrap intervals, p-values, multiplicity claims or favorable
   threshold selection. Report the whole 25-setting surface, including
   unfavorable/censored results, with no threshold cherry-picking.
5. For each original first-pass selection (32 rows), report the original
   budget, correct-count slack `correct-(teacher_correct-5)`, raw NMSE slack
   `.08-nmse`, number of correct predictions that would have to be lost to
   fail the accuracy gate (`count_slack+1`), and whether its selected budget
   changes under one-error gate tightening (allow4) or loosening (allow6) at
   NMSE .08. These are deterministic gate margins, not uncertainty estimates,
   sample-deletion analysis, prediction resampling or new validation evidence.
6. Reproduce the accepted original geometric ratio and 11/4/1 direction
   counts as a guardrail. Preserve the accepted CI as historical context only.

## Deliverables and end condition
A dependency-free Python scalar-analysis script; synthetic boundary,
incompleteness, duplicate and censoring tests; exact portable scalar input;
25-setting CSV/JSON with complete per-seed selections; 32-row selection-margin
CSV; plain-English report of the full surface and scientific limits; independent
recomputation against the same fixed inputs. Missing or inconsistent input is an error, never a reason to replace or omit
data. Original artifacts remain unchanged.
