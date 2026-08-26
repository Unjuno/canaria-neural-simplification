# Canaria documentation index

This directory contains current public-facing documentation and preserved historical research artifacts. Do not infer current project status from file names alone.

## Start here — authoritative current documents

1. `../README.md` — scoped project statement and publication state.
2. `INDEPENDENT_REREVIEW_2026-08-26.md` — independent `KEEP / EDIT / REMOVE / INVALIDATE` decision ledger.
3. `CLAIMS_AND_EVIDENCE.md` — authoritative current public claim registry.
4. `CORE_DISCOVERY.md` — central operational compositional-simplification finding.
5. `PUBLICATION_NOTES.md` — publication-safe wording and evidence tiers.
6. `phase2/README.md` — Phase 2 precision/quantization correction boundary.
7. `RELEASE_CHECKLIST.md` — completed quality/release gate record.
8. `../STATUS.md` — current repository state.

## Scientific evidence and boundaries

- `CROSS_FAMILY_COMPOSITION_REPLICATION.md` — SmallViT direct replication, including the test-isolation caveat found in re-review.
- `CORE_DISCOVERY_REPLICATION_DIGITS.md` — residual-MLP exact-budget direct replication.
- `TRAINING_TIME_CONSOLIDATION.md` — G7–G17 training-time mainline.
- `LATE_STAGE_FINDINGS.md` — G18–G27 mechanism/controller results.
- `NEGATIVE_RESULTS.md` — valid failed hypotheses, boundaries, and distinction from invalidated evidence.
- `REPRODUCIBILITY.md` — evidence/reproduction policy and public-runner boundaries.
- `TERMINOLOGY.md` — current definitions.
- `FAQ.md` — interpretation boundaries and common questions.

Machine-readable evidence lives under `../results/`.

## Systems / application layer

- `APPLICATIONS.md` — application directions separated by evidence status.
- `RUNTIME_POC.md` — bounded small-model CPU serialization/materialization/direct-execution PoC.

Measured systems evidence is intentionally narrow: smaller serialized artifact and lower measured CPU batch-128 inference latency in one small PoC; meaningful host-RAM reduction was not demonstrated; GPU/LLM/energy/general runtime gains are not established.

## Publication / stopping state

The 2026-08-26 independent re-review is complete. The frozen `v0.2.0-public-snapshot` tag/release preserves the pre-Phase-2 baseline, PR #7 was squash-merged after the release boundary was established, and stale research branches were removed. `main` is the sole active branch.

Current `main` is the reviewed post-snapshot public baseline. Future science should begin as a new question/phase rather than extending the old publication-closure sequence.

## Historical / frozen documents

Read `HISTORICAL_INDEX.md` before treating old planning files as current guidance. Historical “next experiment” text is preserved for provenance and is not automatically current project instruction.

## Preservation principle

Do not rewrite old protocols or evidence-producing records to match later theory. Add explicit current interpretation and invalidation/correction records around them. Invalidated evidence remains provenance, not scientific support.
