# Phase 2 precision × composition evidence index

This directory records the first post-v0.2.0 Canaria research phase. It is intentionally isolated on `research/phase2-precision-quantization`; the frozen v0.2.0 snapshot on `main` is not modified by this branch until the release/tag boundary is explicitly crossed.

## Evidence layers

### 1. Protocol locks — pre-result decision rules

- `phase2a/PROTOCOL_LOCK.json`
- `phase2b/PROTOCOL_LOCK.json`
- `phase2c/PROTOCOL_LOCK.json`

These files preserve the hypotheses, fresh-seed sets, endpoint criteria, and PASS/FAIL/UNCERTAIN rules used before result interpretation.

### 2. Recorded observations

#### Phase 2A

The original per-seed JSON objects are normalized into six precision-specific CSV files:

- `grid_32bit.csv`
- `grid_12bit.csv`
- `grid_8bit.csv`
- `grid_6bit.csv`
- `grid_4bit.csv`
- `grid_3bit.csv`

Each file contains all 8 fresh seeds × 7 parameter budgets = 56 grid observations. Across six precisions, this is **336 recorded budget-grid observations**. Columns retain validation NMSE, validation utility, pass/fail status, coded size, scale counts, selected-endpoint flags, and test accuracy only where the original validation-only selection exposed it.

`phase2a/summary.json` contains the predeclared endpoint aggregation.

#### Phase 2B

- `phase2b/seeds_31100_31107.json` — all 8 recorded fresh-seed outputs.
- `phase2b/summary.json` — capacity-rescue decision and max-budget aggregates.

#### Phase 2C

- `phase2c/seeds_31200_31207.json` — all 8 recorded fresh-seed outputs.
- `phase2c/summary.json` — scale-granularity decision and aggregates.

### 3. Portable public runners

See `../../../scripts/phase2/precision_composition/`.

The public runners are portability refactors of the experiment logic. They are **not claimed to be byte-identical historical scripts**, because the original working scripts for 2B/2C imported helper code through local `/mnt/data/...` paths.

A direct portability check was performed for Phase 2C seed `31200`: the refactored runner produced an exactly equal parsed JSON object to the recorded source result.

## Source-bundle provenance

The local handoff/evidence archive used to prepare this branch has SHA256:

`55fd1d94f63773f888a59074227cffc3e0814abace5717f6e6e84b340121cb6c`

See `SOURCE_BUNDLE.sha256`.

The original archive itself is not stored in this Git branch. A binary-upload attempt through the connector was detected as truncated by a size audit and was immediately removed; the repository therefore uses text-native evidence files instead of pretending that incomplete binary provenance is valid.

The source archive also contained transient `__pycache__` files. Those are intentionally excluded from the public research layout.

## Interpretation boundary

- Phase 2A 4-bit result: **PASS** for lower composed minimum passing coded size under the locked rule.
- Phase 2B 3-bit capacity-rescue hypothesis: **FAIL**.
- Phase 2C row-wise scale-granularity rescue hypothesis: **PASS**.
- These results are post-snapshot research and must not be back-projected into the frozen v0.2.0 claim registry without an explicit version transition.

Scientific interpretation belongs in `../../../docs/phase2/PRECISION_COMPOSITION.md`.
