# Environment records

This directory contains both preserved historical environment metadata and current reproduction-environment references.

## Current representative reproduction environment

The headline residual-MLP digits reproduction uses:

- `../scripts/reproduce/core_discovery_digits/requirements-pinned-py311.txt`
- `../scripts/reproduce/core_discovery_digits/README.md`

The current CI reference environment is Python 3.11 with NumPy 2.4.6, PyTorch 2.13.0 CPU, and scikit-learn 1.9.0. The CPU PyTorch wheel is installed explicitly from the PyTorch CPU index before the pinned requirements file is applied.

This is a **modern reproduction environment**, not a claim that these exact package versions were used in every historical experiment.

## Historical environment records

Historical contents are under `history/v10/` and include:

- `REPRODUCIBILITY_LIMITS.md` — limitations of the recorded historical environment;
- `current_audit_environment.json` — captured audit-environment metadata;
- `requirements_inferred.txt` — inferred historical dependencies.

Do not edit historical environment records to match a modern machine. If a clean-port reproduction needs different dependencies, record that environment alongside the new reproduction instead.

## Package boundary

Repository-root `pyproject.toml` describes the lightweight reusable package under `src/canaria/`. It intentionally does not declare every research-runner dependency. Experiment-specific dependencies belong with the relevant reproduction runner.