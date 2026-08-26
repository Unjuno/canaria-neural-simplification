# Reproducibility guide

Canaria separates repository integrity, experimental reproduction, and independent scientific replication.

## Evidence classes

- **Confirmatory** — protocol/seed policy/endpoints locked before fresh outcomes were inspected.
- **Exploratory / pilot** — implementation validation or hypothesis generation.
- **Negative / boundary** — a valid tested hypothesis failed or exposed a limitation.
- **Invalidated** — implementation/protocol defect makes the experiment unusable for scientific inference; preserve provenance but do not count it as negative evidence.
- **Reproduction** — rerun of an already-observed condition to validate software/data portability; not new scientific confirmation by itself.

## Repository integrity

Run:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
python tools/audit_repo.py
```

GitHub CI also runs unit tests and `tools/audit_repo.py`. The independent-review branch additionally requires a minimal public residual-MLP runner smoke test before Issue #9 can close.

An integrity PASS means repository syntax, schemas, required files, correction invariants, and selected semantic guards are consistent. It is **not** a claim that every historical experiment is bitwise reproducible.

## Preferred minimal scientific entry point — residual MLP

```bash
python -m pip install numpy torch scikit-learn
python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 \
  --out /tmp/canaria_seed1200.json
```

Recorded seed 1200 selects:

- component-wise: `3072` learned replacement parameters;
- composed: `1536`.

This public runner has the stronger test-isolation pattern: replacements are fitted from training activations, validation selects the minimum passing budget, and test is evaluated only for the selected endpoint.

## SmallViT replication isolation note

The SmallViT protocol locks candidate selection to training-held-out NMSE and validation utility, excluding test accuracy. However, `scripts/replication/vit_compositional.py` records test metrics for every candidate.

Therefore:

- the test metric is **not a selection variable** under the locked rule;
- the test set was **not operationally hidden** during candidate-result generation;
- the primary result remains usable because the code hash, selection rule, and fresh-seed policy were locked before fresh outcomes;
- future runners should delay test evaluation until after selection, as the residual-MLP runner does.

## Portable reproduction — G7 seed 4300

```bash
python -m pip install -r scripts/reproduce/g7_confirmatory/requirements.txt
python scripts/reproduce/g7_confirmatory/run_seed.py \
  --seed 4300 \
  --out g7_seed_4300.json
```

Recorded reproduction environment:

- Python 3.13.5
- PyTorch 2.10.0+cpu
- NumPy 2.3.5
- scikit-learn 1.8.0

The reproduced JSON exactly matched the archived fresh-confirmatory seed-4300 JSON with SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

This is a **software/portability result for an already-confirmatory seed**, not an independent scientific replication.

## Phase 2 reproducibility and invalidation boundary

### Publicly checked-in A–C evidence

Phase 2A–C have protocol/result files and portable runners under:

- `results/phase2/precision_composition/`
- `scripts/phase2/precision_composition/`

The public runners use internal activation domains consistently and do not rely on `/mnt/data` paths.

### Phase 2E invalidation

Phase 2E is `INVALIDATED_IMPLEMENTATION_BUG`: repair used raw `Xt` instead of internal activation `ta[0]`. Because both had width 64, the error was shape-compatible but semantically wrong.

The invalid result remains in correction history and must not support inference.

Authoritative status:

- `results/phase2/precision_composition/CORRECTION_STATUS.json`
- `results/phase2/precision_composition/INVALIDATED_HISTORY.md`

### Later 2D–2O provenance

Not all later raw per-seed artifacts are checked into this Git branch. The correction archive used for the later audit is identified by SHA256:

`1a339be12d7644de534ac77a712307c49ee0c3d9acb28c8a3532883edca3dab7`

Do not describe later 2D–2O results as fully public portable reproductions merely because their correction status is recorded here.

## Test-set discipline

For new confirmatory runners:

1. fit/train on training or calibration data only;
2. choose candidates using declared validation/holdout criteria;
3. freeze the selected candidate/budget;
4. only then evaluate final test outcomes.

Recording test metrics for all candidates is weaker practice even if a locked selection rule formally excludes them.

## Statistical unit

Repeated spans or checkpoints inside one trained model are correlated. Unless another hierarchy is preregistered, use the independently initialized **training seed/model** as the inferential unit. Prefer paired seed analysis, seed-cluster bootstrap, or leave-one-seed-out evaluation over naive event-level intervals.

## Cost/accounting terminology

Keep distinct:

- learned replacement parameter count;
- optimizer updates / parameter-update proxy;
- nominal quantized bit count;
- scale/metadata bytes;
- serialized bytes;
- FLOPs;
- wall-clock latency;
- energy;
- RAM/VRAM.

A reduction in one does not prove reduction in the others.

Custom 2/3/4/12-bit research quantizers must not be relabeled FP4/FP8 unless a hardware datatype implementation actually exists.

## Runtime PoC boundary

The G7 runtime PoC is one seed, CPU only, small model, batch-128 workload. It supports the recorded serialized-size and CPU-inference observations. It does not establish meaningful host-RAM reduction, GPU/VRAM/energy benefits, or large-model generalization.

## Historical environment limits

Early experiments did not preserve a complete exact package lock. Do not backfill unknown metadata as if it were contemporaneously recorded. Qualitative/aggregate reproduction may be the appropriate target for older phases.

Historical code may contain environment-specific paths. Preserve such code as provenance; add portable runners instead of silently rewriting history.

## Current publication-quality gate

The 2026-08-26 independent re-review is tracked by Issue #9 and `INDEPENDENT_REREVIEW_2026-08-26.md`.

Before that issue closes, the reviewed branch must have:

1. corrected public claims;
2. explicit Phase 2E invalidation history;
3. minimal public-runner smoke PASS;
4. `python tools/audit_repo.py` PASS;
5. GitHub `repository-audit` PASS;
6. root public-surface status updated to the actual completed-review state.

Issues #2 and #3 are historical completed work, not current closure blockers. PR #7 and the v0.2.0 release/tag boundary remain separate repository-state gates after Issue #9.
