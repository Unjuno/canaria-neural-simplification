# Reproducibility Limits

## Important
The historical archive does not contain a complete environment lockfile for every past run. Therefore **bitwise reproduction of historical runs is not guaranteed**.

`current_audit_environment.json` describes the environment used to audit the archive on 2026-08-22; it is not necessarily the original experiment environment.

## What can currently be verified
- Stored Python scripts can be statically syntax-checked.
- Raw CSV/JSON/Markdown evidence is preserved for many runs.
- The historical key-result verification mapped the principal reported values back to raw tables.
- File manifests/SHA256 checks were used on the handoff archives.

## What cannot be reconstructed exactly
- Exact torch/numpy/scikit-learn versions for every historical run.
- CPU/GPU, BLAS/CUDA/cuDNN, and determinism flags for all runs.
- Exact script SHA ↔ raw output provenance for some early exploratory runs.
- Every historical dataset-split hash.

## Remediation for future runs
New work should emit `run_metadata.json` conforming to `schemas/run_metadata_schema.json` alongside every result directory. Historical evidence should not be rewritten to pretend these fields were known.