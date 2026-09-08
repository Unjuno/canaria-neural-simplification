# Canaria public research snapshot

This file defines how to read the frozen v0.2.0 public-snapshot baseline together with the independently reviewed post-snapshot state on `main`.

## Read order

1. `../README.md` — scoped project statement and publication state.
2. `../STATUS.md` — current reviewed public-baseline state.
3. `../REPOSITORY_LAYOUT.md` — repository/evidence lifecycle and branch conventions.
4. `INDEPENDENT_REREVIEW_2026-08-26.md` — independent re-review decision ledger.
5. `CLAIMS_AND_EVIDENCE.md` — authoritative current claim registry.
6. `CORE_DISCOVERY.md` — central operational compositional-simplification claim.
7. `CROSS_FAMILY_COMPOSITION_REPLICATION.md` — SmallViT direct replication and isolation caveat.
8. `CORE_DISCOVERY_REPLICATION_DIGITS.md` — residual-MLP exact-budget replication.
9. `TRAINING_TIME_CONSOLIDATION.md` and `LATE_STAGE_FINDINGS.md` — training-time evidence.
10. `phase2/README.md` — post-snapshot precision/quantization corrections.
11. `NEGATIVE_RESULTS.md` — valid negative evidence versus invalidated evidence.
12. `REPRODUCIBILITY.md` — integrity/reproduction policy.
13. `PUBLICATION_NOTES.md` — publication-safe wording.
14. `RELEASE_CHECKLIST.md` — completed quality/release gate record.
15. `HISTORICAL_INDEX.md` — preserved historical material.

## Version boundary

The immutable reference baseline is tag `v0.2.0-public-snapshot`, pointing to commit:

`556dce21c7a5516a16780cb28d528d1ff3968e53`

GitHub release title:

`Canaria v0.2.0 — Public Research Snapshot`

Post-v0.2.0 evidence and corrections were independently re-reviewed, then squash-merged through PR #7. They live on current `main`; they are not back-projected into the frozen tag.

## Snapshot interpretation

Canaria is an auditable research record, not a production-ready compression library.

The retained core statement is operational:

> under explicit task distributions, replacement grammars, and passing criteria, some learned spans in the tested networks admit smaller task-preserving replacements when fitted as composed input-output functions than when simplified at implementation-component boundaries.

The repository does **not** claim mathematical/Kolmogorov complexity reduction or universal Transformer/LLM behavior.

## Direct architecture-family evidence

### Residual MLP — strongest matched-budget public runner

Fresh seeds `1200–1207` used exact learned replacement-parameter matching at every budget point.

- component-wise mean minimum passing budget: `3584`;
- composed mean minimum passing budget: `1728`;
- composed smaller: `8/8`;
- geometric budget ratio: `0.4823×`;
- selected-budget test-accuracy difference: `+0.583` percentage points, bootstrap95 `[+0.306,+0.806]` pt.

Validation selects the endpoint; test evaluation follows selection.

The 2048-parameter joint-factorized control is descriptive/mechanistic secondary, not a confirmatory causal decomposition.

### Small Vision Transformer

Under the locked rule across 8 fresh eligible seeds:

- component-wise selected replacement: `9808` parameters;
- composed selected replacement: `4904–5424`;
- mean ratio: `0.51988`;
- bootstrap95 `[0.50634,0.53926]`;
- composed smaller: `8/8`.

Independent re-review found an important isolation boundary: the selection criterion excludes test accuracy, but the runner records test metrics for every candidate. Therefore say **“test was not a selection variable”**, not “test remained operationally hidden until after selection.”

## Training-time evidence boundary

The small real-text character-LM program is retained with evidence tiers explicit:

- G7 primary PASS is progressive versus early/late one-shot;
- small-from-start and large-reference differences are secondary observations;
- G15/G17 support an intervening-learning/recontracting interpretation under the tested schedules;
- G18 supports the specific tested deadline-aware controller comparison, not a universal controller rule;
- G21 remains a valid failure;
- G27 remains exploratory without a Pareto claim.

Compiler cost in these experiments is generally a proxy, not measured hardware energy/FLOPs/wall-clock equivalence.

## Phase 2 post-snapshot correction

Phase 2A–C remain usable under their declared residual-MLP quantizer/accounting.

Phase 2E is **`INVALIDATED_IMPLEMENTATION_BUG`** and `DO_NOT_USE_FOR_INFERENCE` because repair used raw `Xt` where the replacement was defined on internal activation `ta[0]`; equal width 64 hid the semantic error.

The invalid result is retained as history, not negative scientific evidence. Phase 2I's RNG causal explanation is retracted. Phase 2O did not confirm a reliable composed repair-sample advantage.

Not all later 2D–2O raw per-seed artifacts are checked into Git. Their correction archive is identified by SHA256:

`1a339be12d7644de534ac77a712307c49ee0c3d9acb28c8a3532883edca3dab7`

See `phase2/README.md` and `../results/phase2/precision_composition/INVALIDATED_HISTORY.md`.

## Reproduction boundary

A self-contained G7 seed-4300 runner exactly matched the archived confirmatory JSON in its recorded environment:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

This is reproduction/portability evidence for an already-confirmatory seed, not a new independent scientific replication.

## Runtime PoC boundary

One small CPU-only PoC reported:

- serialized artifact + manifest: `110,093 → 54,646 bytes`;
- batch-128 CPU inference mean: `47.05 → 23.11 ms`;
- meaningful host-RAM reduction: **not demonstrated**.

Do not generalize this to GPU/VRAM/energy/large models or universal runtime improvement.

## Closure and later-work state

The v0.2.0 publication sequence is complete:

1. Issue #9 independent scientific/public-surface re-review completed and closed;
2. frozen `v0.2.0-public-snapshot` tag/release created at the designated baseline;
3. reviewed PR #7 squash-merged;
4. stale publication-era research branches removed;
5. current-main `repository-audit` used as the final integrity check, with the closing run recorded in Issue #5.

Current `main` remains the reviewed public baseline. Later scientific work may exist on isolated research branches or draft PRs, but it does not update this snapshot or the public claim registry until separately reviewed and merged.
