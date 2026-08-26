# Canaria documentation index

This directory contains reviewed interpretation, locked phase records, current hardening documents, and preserved historical research artifacts. Do not infer current project status from file names or version numbers alone.

Repository-wide path and evidence-lifecycle conventions are in `../REPOSITORY_LAYOUT.md`.

## Start here — current authoritative documents

1. `../README.md` — scoped project statement and pre-announcement state.
2. `../STATUS.md` — current research/readiness status.
3. `ANNOUNCEMENT_READINESS.md` — active announcement gate and blockers.
4. `CLAIMS_AND_EVIDENCE.md` — reviewed claim registry for the current baseline.
5. `INDEPENDENT_REREVIEW_2026-08-26.md` — independent `KEEP / EDIT / REMOVE / INVALIDATE` decision ledger.
6. `CORE_DISCOVERY.md` — central operational compositional-simplification finding.
7. `PUBLICATION_NOTES.md` — communication-safe wording and evidence tiers; not approval to announce.
8. `phase2/README.md` — Phase 2 precision/quantization correction boundary.
9. `REPRODUCIBILITY.md` — evidence/reproduction policy and current pinned-environment gate.

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

- `phases/` — locked/versioned phase protocols and result narratives. These are evidence records, not automatically current headline claims.
- `history/` — preserved handoff/theory material from earlier snapshots.
- `HISTORICAL_INDEX.md` — classification of older planning/status documents that remain at stable paths for provenance.
- `DATA_DICTIONARY.md` — definitions useful for historical result tables.
- `PUBLIC_SNAPSHOT.md` — interpretation of the frozen historical v0.2.0 snapshot.
- `RELEASE_CHECKLIST.md` — historical v0.2.0 checklist, not the current readiness gate.

Historical “next experiment” text is preserved for provenance and is not automatically current project instruction.

## Current readiness / branch state

The repository is **pre-announcement**. `main` is the reviewed evidence baseline used for hardening, not an announcement certificate.

Later research may exist on separate branches or draft PRs. Such work is not part of the headline claim set until separately reviewed and merged. A passing locked experiment is necessary evidence, but does not automatically trigger a claim update or announcement.

Use `ANNOUNCEMENT_READINESS.md` / Issue #13 for the current gate rather than the historical v0.2.0 release checklist.

## Preservation principle

Do not rewrite old protocols, locked results, or evidence-producing records to match later theory. Add explicit current interpretation and invalidation/correction records around them. Invalidated evidence remains provenance, not scientific support.