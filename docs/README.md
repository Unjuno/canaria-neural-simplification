# Canaria documentation index

This directory contains current public-facing documentation, locked phase records, and preserved historical research artifacts. Do not infer current project status from file names or version numbers alone.

Repository-wide path and evidence-lifecycle conventions are in `../REPOSITORY_LAYOUT.md`.

## Start here — authoritative current documents

1. `../README.md` — scoped project statement and publication state.
2. `../STATUS.md` — current reviewed public-baseline state.
3. `CLAIMS_AND_EVIDENCE.md` — authoritative current public claim registry.
4. `INDEPENDENT_REREVIEW_2026-08-26.md` — independent `KEEP / EDIT / REMOVE / INVALIDATE` decision ledger.
5. `CORE_DISCOVERY.md` — central operational compositional-simplification finding.
6. `PUBLICATION_NOTES.md` — publication-safe wording and evidence tiers.
7. `phase2/README.md` — Phase 2 precision/quantization correction boundary.
8. `REPRODUCIBILITY.md` — evidence/reproduction policy and public-runner boundaries.

## Scientific evidence and boundaries

- `CORE_DISCOVERY_REPLICATION_DIGITS.md` — residual-MLP exact-budget direct replication.
- `CROSS_FAMILY_COMPOSITION_REPLICATION.md` — SmallViT direct replication, including the test-isolation caveat found in re-review.
- `TRAINING_TIME_CONSOLIDATION.md` — G7–G17 training-time mainline.
- `LATE_STAGE_FINDINGS.md` — G18–G27 mechanism/controller results.
- `NEGATIVE_RESULTS.md` — valid failed hypotheses, boundaries, and distinction from invalidated evidence.
- `TERMINOLOGY.md` — current definitions.
- `FAQ.md` — interpretation boundaries and common questions.

Machine-readable evidence lives under `../results/`; use `../results/README.md` to distinguish headline evidence, reproduction evidence, correction history, and extended historical phase results.

## Systems / application layer

- `APPLICATIONS.md` — application directions separated by evidence status.
- `RUNTIME_POC.md` — bounded small-model CPU serialization/materialization/direct-execution PoC.

Measured systems evidence is intentionally narrow: smaller serialized artifact and lower measured CPU batch-128 inference latency in one small PoC; meaningful host-RAM reduction was not demonstrated; GPU/LLM/energy/general runtime gains are not established.

## Protocol and history layer

- `phases/` — locked/versioned phase protocols and result narratives. These are evidence records, not automatically current public claims.
- `history/` — preserved handoff/theory material from earlier snapshots.
- `HISTORICAL_INDEX.md` — classification of older planning/status documents that remain at stable paths for provenance.
- `DATA_DICTIONARY.md` — definitions useful for historical result tables.

Historical “next experiment” text is preserved for provenance and is not automatically current project instruction.

## Publication and branch state

The 2026-08-26 independent re-review and the v0.2.0 publication sequence are complete. The frozen `v0.2.0-public-snapshot` tag/release preserves the designated baseline, and current `main` is the reviewed post-snapshot public baseline.

Post-publication research may exist on separate branches or draft PRs. Such work is not part of this authoritative public surface until separately reviewed and merged. Consult open GitHub PRs/issues for work in progress rather than inferring it from `main` documentation.

## Preservation principle

Do not rewrite old protocols, locked results, or evidence-producing records to match later theory. Add explicit current interpretation and invalidation/correction records around them. Invalidated evidence remains provenance, not scientific support.
