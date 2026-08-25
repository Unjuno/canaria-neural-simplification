# Reproducibility guide

Canaria contains both cleaned reusable code and provenance-preserving historical evidence scripts. Reproducibility therefore has two levels:

1. **repository integrity** — files, schemas, syntax, and machine-readable summaries are internally consistent;
2. **experimental reproduction** — a specific historical/confirmatory experiment can be recreated under the documented data, seed, and environment assumptions.

## Quick integrity check

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/audit_repo.py
```

CI runs unit tests plus `tools/audit_repo.py` on pushes and pull requests.

## Current public-snapshot invariants

The audit requires the public entry points and evidence manifests to remain present, including:

- `README.md`
- `STATUS.md`
- `docs/CORE_DISCOVERY.md`
- `docs/CLAIMS_AND_EVIDENCE.md`
- `docs/TRAINING_TIME_CONSOLIDATION.md`
- `docs/LATE_STAGE_FINDINGS.md`
- `docs/NEGATIVE_RESULTS.md`
- `docs/APPLICATIONS.md`
- `docs/OPEN_QUESTIONS.md`
- `results/training_time/summary.json`
- `results/training_time/protocol_manifest.json`
- `results/training_time/late_stage_summary.json`

The audit also checks Python syntax and parses repository JSON/CSV files. It is an integrity check, not a claim that every historical experiment is bitwise reproducible from one command.

## Historical environment limits

Early experiments did **not** preserve a complete exact package lock. Reproduce qualitative/aggregate behavior first; do not expect bitwise identity from the oldest scripts.

The historical audit environment is preserved under `environment/history/v10/`. Do not backfill unknown metadata as if it had been recorded contemporaneously.

## Evidence classes

- **Confirmatory** — conditions/endpoints/seed policy locked before fresh-seed outcome inspection.
- **Independent holdout** — selected condition retested without reselection.
- **Exploratory / pilot** — implementation validation or hypothesis generation.
- **Negative / boundary** — failed hypothesis retained explicitly.

Do not promote exploratory runs to confirmatory after results are known.

## Statistical unit

Repeated spans or fit checkpoints within one trained network are correlated. Unless a different hierarchy is preregistered, the independently initialized **training seed/model** is the inferential unit. Prefer seed-cluster bootstrap, paired seed analysis, or leave-one-seed-out evaluation over naive event-level intervals.

## Matched continuation controls

Repair/recontracting experiments should compare the compiled candidate against a matched uncompiled/teacher continuation receiving the same task-training budget and minibatch schedule whenever the question concerns recovery over time.

## Test-set isolation

Autonomous controller decisions must not use final test outcomes. Training/calibration/validation data used for commit decisions should be declared separately from final evaluation data.

## Protocol integrity

Major confirmatory phases should preserve:

- protocol lock or equivalent preregistration artifact;
- fresh seed range;
- primary endpoint and decision rule;
- code/script hash when available;
- result summary hash;
- known deviations or metadata limitations.

For G18–G26, public headline values and protocol/result SHA256 values are indexed in `results/training_time/late_stage_summary.json`.

## Historical blind-map rule

The original Phase-A blindness procedure was:

1. train eligible baselines;
2. evaluate simplification candidates without computing Canary;
3. save and hash-lock the Stage-1 table;
4. compute Canary only after that lock;
5. join sensor and simplification tables.

The original locks remain under `results/phaseA_v11/`.

## Storage terminology

Keep these distinct:

- `core bytes` — compiled replacement only;
- `whole-network bytes` — all model components charged by the declared codec;
- nominal bit count — bookkeeping estimate;
- entropy/ideal code length — not necessarily a real file;
- serialized bytes — an actual materialized artifact.

The 9,926-byte v19 endpoint is an exact round-tripped whole-model serialization under its declared codec, not a codec-independent lower bound.

## Precision terminology

Custom 2/3/4/12-bit experiments are research quantizers unless a hardware datatype is explicitly used. Do not relabel them as FP4/FP8 without implementation evidence.

## Recommended reproduction order for a new contributor

1. Run the repository audit and unit tests.
2. Read `CORE_DISCOVERY.md`, `CLAIMS_AND_EVIDENCE.md`, and `NEGATIVE_RESULTS.md`.
3. Reproduce one cleaned reusable codec/unit-test path.
4. Reproduce one historical residual-CNN confirmatory result if studying the original compositional phenomenon.
5. Reproduce one training-time consolidation result (G15/G17 or later) if studying recontracting.
6. Only then add a new architecture/task or deployment proof-of-concept.

## Current closure target

The public snapshot would benefit from an externally runnable clean-repository reproduction of one representative confirmatory pipeline. This is a **reproducibility closure task**, not a request to reopen broad experiment search.

Historical evidence scripts may contain environment-specific paths. Do not silently rewrite those scripts; add a portable runner under cleaned source code and validate equivalence against the preserved evidence where possible.
