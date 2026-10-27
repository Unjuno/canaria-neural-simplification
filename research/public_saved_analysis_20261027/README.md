# Portable saved-result scalar analysis

This is a **new analysis-only implementation**, not an original executed training,
measurement or orchestration runner. It recalculates conclusions from recorded
scalar tables. It does not establish independent experimental reproduction,
validate raw dataset/model tensors, or supply new scientific evidence.

## Run

Python 3.11+ and NumPy 2.3.5. No torch, datasets, credentials or network are used.
From this directory, with the existing checkpoint in `../public_checkpoint_20261027`:

```sh
python -m unittest -v
python check_checkpoint.py ../public_checkpoint_20261027
python analyze_saved.py matched ../public_checkpoint_20261027/research/digits_matched_regime/results/SUMMARY.json
python analyze_saved.py grid ../public_checkpoint_20261027/research/digits_joint_budget_grid/results/evidence/SUMMARY.json
python analyze_saved.py storage ../public_checkpoint_20261027/research/selected_whole_storage/results/ROWS.csv
python analyze_saved.py latency ../public_checkpoint_20261027/research/selected_latency/results/RAW.jsonl
```

All inputs are caller-specified; examples use relative paths. Outputs go to stdout.
The checker requires exact equality for relevant saved summary fields, including
bootstrap endpoints, per-seed minima and medians. It ignores unrelated metadata.
CSV derived ratio columns are deliberately ignored and recomputed from numerators.
No models, arrays, images or datasets are required or included.

## Input contracts and method

* Matched: JSON `all_rows`: exactly IDs 0–15, COMPLETE, C/J/L arms. Each arm
  records `n=270`, integer `correct`, nonnegative finite `nmse`, Boolean
  `quality_pass`; seed records teacher_correct. Optional accuracy must equal
  correct/270. Quality = nmse <= .08 AND correct/270 >= teacher_correct/270−.02.
  Mean log2(J/L) uses max(nmse,1e-12), and paired accuracy (J−L)/270.
  100,000 bootstrap draws, NumPy default_rng seed 271027042, sampled indices
  shape (100000,16), mean then linear quantiles [.025,.975]. Same indices for
  both outcomes. Feature gate CI upper<0; accuracy gate CI lower>−.02.
* Grid: JSON `all_rows`, same all-16 contract. Exactly 14 distinct C/J ×
  [512,1024,1536,2048,3072,4096,6144] cells/seed; only 2048 `reused=true`.
  Recheck quality from counts and nmse. Record first passing grid budget, pass
  counts and any later failure following a pass. If any seed/arm is censored,
  report coverage but **no cohort ratio or complete-case rescue**. Otherwise
  bootstrap paired log2(C/J) using seed 2710276144 and the same draw/quantile
  convention; exponentiate mean and CI. This is not a global capacity minimum.
* Storage: CSV, all 16 IDs and positive recorded whole-model parameter counts,
  stored bytes and deflated bytes for C/J/teacher. Recalculate all nine paired
  ratio summaries (geomean, min/max and direction counts). This does not reopen
  archives or prove deployed memory savings. Parameter/byte inputs are integers.
* Latency: JSONL of exactly 2,880 existing timing blocks, all 16 IDs × batches
  1/128 × three arms × rounds 0–29. Require unique cell/round identities and
  positions 0–2 per round, 20 iterations, positive finite timings and exact
  total_ns/20 arithmetic. Median over all 30 blocks without outlier exclusions,
  then six paired ratio summaries. No timer, forward pass or benchmark runs.

Malformed/incomplete inputs fail rather than silently dropping observations.
Missing fields raise errors, with CLI exit 2. Grid censoring is a legitimate
output distinct from malformed data. Synthetic tests cover missing fields,
nonfinite/negative values, duplicates, count/quality inconsistencies, all-16
coverage, censoring, nonmonotonicity and timing arithmetic.

## Scope

Matched/grid conclusions remain conditional on reused digits and teacher/split
choices, not fresh-data generalization or objective-only causality. Storage is
whole-archive bookkeeping. Timing remains shared-host eager CPU synthetic-input
steady-state measurement, not a portable hardware or depth-only causal claim.

`SOURCE_PROVENANCE.json` identifies original algorithm sources by repository-relative
path, commit and SHA256. `TRANSFORMATION.md` describes changes. The source hash is
not the hash of this newly written module. Public scalar inputs are not copied here.

## Separate post-hoc threshold sensitivity

The adjacent [saved-scalar threshold analysis](../threshold_robustness_20261027/README.md)
examines 25 prespecified sensitivity settings on the same already-seen 224-cell
validation grid. Its [report](../threshold_robustness_20261027/REPORT.md) retains
all settings, suppresses cohort ratios when a seed/arm has no passing candidate,
and reports selection margins. It does not alter this module's original quality
criterion, selections, results, or historical 29-field / 12-test validation record.
It is descriptive post-hoc analysis, not new data, model execution or independent
experimental reproduction. Run it using its own standard-library-only instructions.
