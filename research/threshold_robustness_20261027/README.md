# Post-hoc saved-scalar threshold robustness

Start with [REPORT.md](REPORT.md). This is a finite descriptive reanalysis of
already-seen validation results, **not a new experiment or confirmation**.
Original criteria and accepted results remain unchanged.

## Reproduce

Python standard library only; no package installation, dataset, model,
credentials or network is required. From this directory:

```sh
python -m unittest -v
python analyze_thresholds.py SCALAR_INPUT.json reproduction
```

The output directory must not exist. Compare all five generated files with
`results/`; they are deterministic. Missing or malformed input raises an error;
no-pass cases are explicitly censored, never dropped to rescue a cohort ratio.

Optionally reconstruct the portable input from the existing public checkpoint:

```sh
python prepare_input.py ../public_checkpoint_20261027/research/digits_joint_budget_grid/results/evidence/SUMMARY.json SOURCE_PROVENANCE.json scalar-input-reproduction.json
```

The public source and resulting input hashes are checked; existing output files
are never overwritten. The historical original-source hash in `SCALAR_INPUT.json`
is distinct from the public source hash: administrative metadata was removed in
that earlier public export, but all selected scalar fields are exactly equal.

## Files and interpretation

- [PROTOCOL.md](PROTOCOL.md): public scientific extract of the fixed post-hoc plan.
- `SCALAR_INPUT.json`: all 16 seeds and 224 scalar cells, plus original references.
- `analyze_thresholds.py`: unchanged scalar-analysis implementation.
- `test_analyze_thresholds.py`: 14 synthetic tests; no real model experiments.
- `prepare_input.py`: public-source projection with hash checks.
- `results/SURFACE.csv`: all 25 settings, coverage and censoring.
- `results/FIRST_PASSES.csv`: all 400 paired seed-setting selections.
- `results/PASS_COUNTS.csv`: all 175 budget-setting count pairs.
- `results/SELECTED_MARGINS.csv`: all 32 original selection margins.
- `results/RESULTS.json`: full machine-readable result.
- [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) and
  [TRANSFORMATION.md](TRANSFORMATION.md): source identity and export changes.
- `FILES_SHA256.json`: public-package hashes, excluding only the manifest itself.

Twenty settings have all 16 pairs and C/J ratios below 1; the other five are
censored, **not successful robustness results**. Individual selections remain
fragile. The 25 settings reuse the same cohort and are not independent trials.
No new confidence intervals or significance claims are supplied. The original
interval retained in the input is historical context, not an interval for any
alternative threshold. See the report for all limitations.
