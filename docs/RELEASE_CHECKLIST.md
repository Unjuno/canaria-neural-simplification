# Public snapshot release checklist

This checklist is for freezing the current repository as a public research snapshot. It is intentionally operational and should not reopen broad experiment search.

## Repository state

- [x] README centered on compositional simplification rather than generic compression.
- [x] Current claim registry separated from historical documents.
- [x] Positive, negative, exploratory, confirmatory, reproduction, systems-PoC, and fresh replication evidence distinguished.
- [x] G18–G26 late-stage evidence indexed; G21 failure preserved.
- [x] Portable exact reproduction added for G7 fresh confirmatory seed 4300.
- [x] Minimal CPU runtime/materialization PoC added with explicit RAM/generalization boundaries.
- [x] Direct SmallViT component-wise-versus-composed replication completed under a locked fresh protocol.
- [x] `docs/README.md` and `HISTORICAL_INDEX.md` separate current from frozen historical material.
- [x] Repository audit enforces public-snapshot invariants.
- [x] GitHub Issue #1 closed as completed.
- [x] GitHub Issue #2 closed as completed.
- [x] GitHub Issue #3 closed as completed.
- [x] No scientific experiment remains required for v0.2.0 at the present claim scope.

## Scientific closure decision

The current snapshot supports the scoped statement:

> Canaria identifies and characterizes task-conditioned compositional simplification in learned neural computation under declared task distributions, replacement grammars, and accounting rules.

The core phenomenon is supported by the original residual-CNN confirmatory program and by a fresh direct Small Vision Transformer replication. In the SmallViT experiment, the minimum passing composed replacement used about **52%** of the component-wise replacement complexity on average, with a paired seed-bootstrap95 ratio **[0.506, 0.539]** and 8/8 fresh seeds favoring composition.

This still does not justify universal Transformer/LLM or mathematical-complexity claims.

## Remaining optional GitHub UI metadata

The connected automation does not expose repository-description or tag/release creation, so these remain manual UI actions if desired.

Suggested repository description:

> Research on task-conditioned compositional simplification, training-time consolidation, and compact functional representations of learned neural computation.

Suggested snapshot tag:

`v0.2.0-public-snapshot`

Suggested release title:

`Canaria v0.2.0 — Public Research Snapshot`

Suggested release summary:

> Public research snapshot centered on task-conditioned compositional simplification, including original residual-CNN evidence and a fresh direct SmallViT replication, training-time consolidation/recontracting, preserved negative results, an exact portable G7 confirmatory reproduction, and a bounded CPU runtime/materialization proof of concept. The snapshot does not claim universal mathematical complexity reduction, large-LLM validity, or general RAM/GPU/runtime gains.

## Before creating a tag/release

1. Verify `main` passes `.github/workflows/ci.yml`.
2. Optionally run the manual `reproduce-g7.yml` workflow.
3. Optionally run the manual `runtime-poc.yml` workflow.
4. Read `PUBLICATION_NOTES.md` and ensure release text uses the same claim hierarchy.
5. Confirm `results/replication/vit_compositional/` remains present and audit-checked.

## Citation

Until a paper exists, cite:

- the exact commit or snapshot tag;
- the relevant protocol/result artifacts or recorded SHA256 values;
- the evidence class for any quoted endpoint.

`CITATION.cff` is the canonical repository citation metadata.

## Stopping rule

After the optional release metadata is set, treat the current repository as a **frozen public research snapshot**. Future scientific work should begin from a new issue/question or a new research phase rather than silently extending the old G-number mainline.
