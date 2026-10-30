# Reusable code layer

`src/canaria/` is deliberately different from `scripts/`:

- `scripts/` preserves evidence-producing historical experiment code, including some original filesystem assumptions.
- `src/canaria/` contains cleaned, dependency-light interfaces intended for reuse in new code.

Current reusable module:

## `canaria.ternary_codec`

Extracts the exact v17 core serialization logic into pure Python:

- five-trits-per-byte ternary packing;
- 18 × 1-of-4 support-pattern packing;
- exact 38-byte fixed v17 stream;
- zero-aware combinadic/enumerative compression and decompression.

The module does **not** train or approximate a model. It only serializes an already selected structured ternary representation, so round-trip decoding must preserve the exact discrete model state.

The package metadata declares no experiment dependency bundle. Installing it does not install PyTorch/scikit-learn or provide research datasets, saved model artifacts, releases or custody records. Historical and current experiment entrypoints remain distinct; choose one through the [reproducibility guide](../docs/REPRODUCIBILITY.md) before preparing its separate environment.

Install the lightweight package from the repository root:

```bash
python -m pip install -e .
python -m unittest discover -s tests -p test_ternary_codec.py -v
```

The command above selects the codec-only tests. The full test discovery also imports the baseline cohort verifier and therefore needs its NumPy dependency; use the separate recorded baseline environment for that suite. This command selection is based on static import inspection, not a newly recorded test run.

Model-specific experimental code remains under `scripts/phases/`; reusable components should be migrated into `src/canaria/` only when the interface can be separated from a historical experiment without changing the original evidence chain.
