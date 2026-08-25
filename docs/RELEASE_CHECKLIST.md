# Public snapshot release checklist

This checklist is for freezing the current repository as a public research snapshot. It is intentionally operational and should not reopen broad experiment search.

## Repository state

- [x] README centered on compositional simplification rather than generic compression.
- [x] Current claim registry separated from historical documents.
- [x] Positive, negative, exploratory, confirmatory, reproduction, and systems-PoC evidence distinguished.
- [x] G18–G26 late-stage evidence indexed; G21 failure preserved.
- [x] Portable exact reproduction added for G7 fresh confirmatory seed 4300.
- [x] Minimal CPU runtime/materialization PoC added with explicit RAM/generalization boundaries.
- [x] `docs/README.md` and `HISTORICAL_INDEX.md` separate current from frozen historical material.
- [x] Repository audit enforces public-snapshot invariants.
- [x] GitHub Issues #1 and #3 closed as completed.
- [x] Issue #2 classified as optional future work rather than a blocker for the current scoped claim.

## Scientific closure decision

The current snapshot deliberately keeps the scientific claim scoped to the tested settings:

> Canaria identifies and characterizes task-conditioned compositional simplification in learned neural computation under declared task distributions, replacement grammars, and accounting rules.

A direct replication of component-wise versus composed-span simplification on a clearly different architecture/task would strengthen a future cross-family generalization or priority claim, but it is **not required** for the present snapshot wording.

GitHub Issue #2 is therefore future work, not unfinished closure work for v0.2.0.

## Suggested GitHub UI metadata

The connected automation currently does not expose repository-description or tag/release creation, so these remain manual UI actions if desired.

Suggested repository description:

> Research on task-conditioned compositional simplification, training-time consolidation, and compact functional representations of learned neural computation.

Suggested snapshot tag:

`v0.2.0-public-snapshot`

Suggested release title:

`Canaria v0.2.0 — Public Research Snapshot`

Suggested release summary:

> Public research snapshot centered on task-conditioned compositional simplification, training-time consolidation/recontracting, preserved negative results, an exact portable G7 confirmatory reproduction, and a bounded CPU runtime/materialization proof of concept. The snapshot does not claim universal mathematical complexity reduction, large-LLM validity, or general RAM/GPU/runtime gains.

## Before creating a tag/release

1. Verify `main` passes `.github/workflows/ci.yml`.
2. Optionally run the manual `reproduce-g7.yml` workflow.
3. Optionally run the manual `runtime-poc.yml` workflow.
4. Read `PUBLICATION_NOTES.md` and ensure release text uses the same claim hierarchy.
5. Keep future cross-family replication separate from the frozen current snapshot; reopen/new-issue it only if a later publication requires stronger external-validity language.

## Citation

Until a paper exists, cite:

- the exact commit or snapshot tag;
- the relevant protocol/result artifacts or recorded SHA256 values;
- the evidence class for any quoted endpoint.

`CITATION.cff` is the canonical repository citation metadata.

## Stopping rule

After the release metadata is set, treat the current repository as a **frozen public research snapshot**. Future scientific work should begin from a new issue/question or a new research phase rather than silently extending the old G-number mainline.
