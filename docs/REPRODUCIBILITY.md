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

## Public portable reproduction: G7 seed 4300

The repository now contains a self-contained portable runner for one representative fresh confirmatory pipeline:

```bash
python -m pip install torch numpy scikit-learn
python scripts/reproduce/g7_confirmatory/run_seed.py \
  --seed 4300 \
  --out g7_seed_4300.json
```

The historical G7 code depended on private `/mnt/data` import paths. The public runner removes those path assumptions while preserving the scientific computation: model definitions, deterministic data-window generation, seed schedule, minibatch order, optimizer settings, learning-rate schedule, compiler budgets, replacement sequence, and metrics are unchanged.

On 2026-08-25 the portable runner was executed in:

- Python 3.13.5
- PyTorch 2.10.0+cpu
- NumPy 2.3.5
- scikit-learn 1.8.0

The complete reproduced JSON exactly matched the archived fresh-confirmatory seed-4300 JSON, with identical SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

Headline test PPL values were:

- large reference: `19.278388330876`
- small from start: `20.49408599251734`
- terminal post-hoc: `19.30211076363844`
- late one-shot: `19.33549169102473`
- early one-shot: `19.164090006166735`
- progressive compute-matched: `18.932213342799887`

See:

- `scripts/reproduce/g7_confirmatory/README.md`
- `results/reproduction/g7_seed4300_report.json`
- `.github/workflows/reproduce-g7.yml`

The workflow is manual (`workflow_dispatch`) so the ~full confirmatory-seed computation is not charged to every routine repository push.

This exact match is a **software/reproducibility result**, not a new independent scientific replication, because seed 4300 was already part of the original fresh G7 confirmatory cohort.

## Current public-snapshot invariants

The audit requires the public entry points and evidence manifests to remain present, including:

- `README.md`
- `STATUS.md`
- `docs/PUBLIC_SNAPSHOT.md`
- `docs/CORE_DISCOVERY.md`
- `docs/CLAIMS_AND_EVIDENCE.md`
- `docs/TRAINING_TIME_CONSOLIDATION.md`
- `docs/LATE_STAGE_FINDINGS.md`
- `docs/NEGATIVE_RESULTS.md`
- `docs/APPLICATIONS.md`
- `docs/OPEN_QUESTIONS.md`
- `docs/PUBLICATION_NOTES.md`
- `docs/TERMINOLOGY.md`
- `docs/FAQ.md`
- `results/training_time/summary.json`
- `results/training_time/protocol_manifest.json`
- `results/training_time/late_stage_summary.json`
- `scripts/reproduce/g7_confirmatory/run_seed.py`
- `results/reproduction/g7_seed4300_report.json`

The audit also checks Python syntax and parses repository JSON/CSV files. It is an integrity check, not a claim that every historical experiment is bitwise reproducible from one command.

## Historical environment limits

Early experiments did **not** preserve a complete exact package lock. Reproduce qualitative/aggregate behavior first; do not expect bitwise identity from the oldest scripts.

The historical audit environment is preserved under `environment/history/v10/`. Do not backfill unknown metadata as if it had been recorded contemporaneously.

The exact G7 seed-4300 reproduction above should not be generalized into a claim that all historical phases are byte-reproducible under the same environment.

## Evidence classes

- **Confirmatory** — conditions/endpoints/seed policy locked before fresh-seed outcome inspection.
- **Independent holdout** — selected condition retested without reselection.
- **Exploratory / pilot** — implementation validation or hypothesis generation.
- **Negative / boundary** — failed hypothesis retained explicitly.
- **Reproduction** — rerun of an already-observed condition to validate software/data portability; not new scientific confirmation by itself.

Do not promote exploratory runs to confirmatory after results are known, and do not count a reproduction seed as a new independent seed.

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
3. Run the portable G7 seed-4300 reproduction or the manual GitHub Actions workflow.
4. Reproduce one historical residual-CNN confirmatory result if studying the original compositional phenomenon.
5. Reproduce G15/G17 or later if studying the staged-vs-direct/recontracting mechanism specifically.
6. Only then add a new architecture/task or deployment proof-of-concept.

## Current closure state

The clean-repository portability gap for one representative confirmatory pipeline is now closed by the G7 seed-4300 runner and exact recorded reproduction.

Remaining research is conditional rather than required for repository closure:

- Issue #2: direct replication of compositional simplification on a different family, only if a stronger generalization/novelty claim is pursued;
- Issue #3: minimal runtime-compilation proof-of-concept, only if deployment claims are pursued.

Historical evidence scripts may still contain environment-specific paths. Do not silently rewrite those scripts; add portable runners and validate equivalence against preserved evidence where possible.
