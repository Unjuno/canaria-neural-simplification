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

Install the lightweight package from the repository root:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Model-specific experimental code remains under `scripts/phases/`; reusable components should be migrated into `src/canaria/` only when the interface can be separated from a historical experiment without changing the original evidence chain.