# Reproducibility guide

Canaria separates repository integrity, experimental reproduction, and independent scientific replication.

## Choose the reproduction route

For the current 16-teacher J/C study, start with the [research assessment](../RESEARCH_CHECKPOINT.md) and [full report](../research-checkpoints/2026-10-27/README.md). The older residual-MLP and G7 commands below reproduce **different historical studies**, not this current pipeline.

| Goal | Prerequisites and entry | Output and evidence boundary |
|---|---|---|
| Recalculate recorded summaries | [Saved-result analysis](../research/public_saved_analysis_20261027/README.md): Python 3.11+, NumPy 2.3.5, and the checked-in scalar checkpoint | Summaries/checks on stdout; no training, model loading, new timing, or dataset/network access |
| Recalculate post-hoc threshold sensitivity | [Threshold analysis](../research/threshold_robustness_20261027/README.md): standard library and bundled scalar input | Five files in a new, absent output directory; preserves censored cases; not independent trials |
| Inspect or plan a fresh technical reproduction | [Matched source](../research/digits_matched_source_20261027/README.md) → [grid source](../research/digits_joint_budget_source_20261027/README.md) → [fresh selection](../research/selected_whole_storage_source_20261027/FRESH_SELECTION.md) → [storage](../research/selected_whole_storage_source_20261027/README.md) → [fresh latency inputs](../research/selected_latency_source_20261027/FRESH_INPUTS.md) → [latency](../research/selected_latency_source_20261027/README.md) | New private artifacts from separately authorized stages; source packages alone do not supply original weights/arrays, approved releases, or completed custody records |

Use each linked package's environment, input, output and provenance contracts; do not substitute historical quickstart outputs for a fresh matched/grid parent. Keep pinned sibling packages together. Saved scalar tables cannot reconstruct missing weights or validation arrays. Source maps and hash checks establish specified byte relationships, not execution authority or independent scientific replication.

**Known fresh-run failure:** the [2026-10-29 attempt](../research-checkpoints/2026-10-29/REPRODUCTION_STATUS.md) failed during checkout preparation, before scientific stages or runtime/resource preflight completed. It produced no new scientific result, and its consumed attempt is not reusable. Current validation status is in [STATUS.md](../STATUS.md); package-local “pending” statements may describe their original packaging date. Recorded synthetic-test passes do not establish fresh end-to-end reproduction.

The remainder preserves historical procedures and general interpretation guidance. Installation commands change an environment; scientific runners train/evaluate models. They are not required for saved-result recalculation and are not instructions to restart a closed attempt.

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

GitHub CI also runs unit tests and `tools/audit_repo.py`. The publication workflow also repeats the complete recorded core cohort. For the historical core publication workflow, Issue #9 was superseded by Issue #13 and `ANNOUNCEMENT_READINESS.md`; this is not a completion gate for the current J/C pipeline.

An integrity PASS means repository syntax, schemas, required files, correction invariants, and selected semantic guards are consistent. It is **not** a claim that every historical experiment is bitwise reproducible.

## Historical residual-MLP scientific entry point

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts/reproduce/core_discovery_digits/requirements-pinned-py311.txt
python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 \
  --out /tmp/canaria_seed1200.json
```

Recorded seed 1200 selects:

- component-wise: `3072` learned replacement parameters;
- composed: `1536`.

This public runner has the stronger test-isolation pattern: replacements are fitted from training activations, validation selects the minimum passing budget, and candidate test is evaluated at selected endpoints; teacher and a prespecified mechanistic control also have test metrics. Test values are not used by budget selection.

## SmallViT replication isolation note

The SmallViT protocol locks candidate selection to training-held-out NMSE and validation utility, excluding test accuracy. However, `scripts/replication/vit_compositional.py` records test metrics for every candidate.

Therefore:

- the test metric is **not a selection variable** under the locked rule;
- the test set was **not operationally hidden** during candidate-result generation;
- the primary result remains usable because the code hash, selection rule, and fresh-seed policy were locked before fresh outcomes;
- future runners should delay test evaluation until after selection, as the residual-MLP runner does.

## Historical portable reproduction — G7 seed 4300

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

## Historical core publication-quality gate

See [ANNOUNCEMENT_READINESS.md](ANNOUNCEMENT_READINESS.md) and [Quickstart](../QUICKSTART.md). That historical core workflow targets Python3.11 and pinned CPU numerical dependencies. A complete environment record is saved alongside the reproduction. [PyTorch documents](https://docs.pytorch.org/docs/stable/notes/randomness.html) that bitwise reproducibility is not guaranteed across versions/platforms. Do not promise CPU/GPU parity from equal seeds.

The historical re-review ledger is retained; new numerical audits do not constitute independent external scientific review. Other historical studies are linked through the [research index](RESEARCH_INDEX.md); the current J/C report uses the routes above.

Fixed update count600 with batch128 already fixes total sampled-example draws at76800. Calibration and validation NMSE can have different denominators; do not subtract them as if they were a common-scale generalization gap. A failed non-inferiority gate need not establish inferiority.

## Unresolved historical core discrepancy

The full-cohort check found a strict mismatch in two version/platform stacks. [Evidence and limitation](REPRODUCTION_DISCREPANCY.md). Do not substitute the one-seed smoke result or a qualitative8/8 agreement for exact numerical reproduction.
