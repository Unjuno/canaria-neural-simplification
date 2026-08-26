# Environment records

This directory documents the **current supported reproduction environment**. Historical environment captures were moved to `../archives/research-history/environment/` so they are not confused with present installation guidance.

## Representative reproduction environment

The headline residual-MLP digits reproduction uses:

- `../scripts/reproduce/core_discovery_digits/requirements-pinned-py311.txt`
- `../scripts/reproduce/core_discovery_digits/README.md`

Current pinned reference:

- Python 3.11
- NumPy 2.4.6
- PyTorch 2.13.0 CPU
- scikit-learn 1.9.0

The CPU PyTorch wheel is installed explicitly from the PyTorch CPU index before applying the pinned requirements file.

This is a modern reproduction target. It is **not** a claim that every historical experiment used these exact versions.

## Historical environments

Earlier inferred/captured environment metadata is preserved under:

`../archives/research-history/environment/history/v10/`

Do not rewrite those records to match a modern machine. If an archived experiment is ported, record the new environment as a separate reproduction provenance record.

## Package boundary

Repository-root `pyproject.toml` describes the lightweight reusable package under `src/canaria/`. It intentionally does not encode every historical research-runner dependency.
