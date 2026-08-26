# Historical v0.2.0 snapshot checklist

This file records the **completed historical v0.2.0 snapshot/review/version-control sequence**. It is not the current announcement-readiness checklist.

For the active gate, use `ANNOUNCEMENT_READINESS.md` and Issue #13.

## Scientific/claim review — Issue #9

- [x] Root README centered on the bounded operational compositional-simplification claim.
- [x] Claim registry separated replacement/description complexity from mathematical/Kolmogorov complexity.
- [x] Residual-MLP direct replication re-reviewed for split isolation, exact learned-budget accounting, seed rules, and statistics.
- [x] SmallViT direct replication re-reviewed; all-candidate test recording disclosed as a weaker isolation boundary even though test is not a locked selection variable.
- [x] Training-time primary/secondary and exploratory/confirmatory boundaries re-reviewed.
- [x] Phase 2E marked `INVALIDATED_IMPLEMENTATION_BUG`; downstream causal interpretations corrected/retracted.
- [x] Phase 2O positive repair-sample-complexity claim removed; result remains `UNCERTAIN`.
- [x] Invalidated evidence preserved as correction history rather than silently erased.
- [x] G7 exact rerun described as reproduction/portability evidence, not an independent scientific replication.
- [x] Runtime PoC restricted to the measured small CPU/storage/inference scope.
- [x] Independent re-review ledger added: `INDEPENDENT_REREVIEW_2026-08-26.md`.
- [x] Issue #9 closed as completed.

## Repository quality gate at the time

The reviewed tree passed GitHub `repository-audit`, including:

- [x] reusable codec unit tests;
- [x] `python tools/audit_repo.py` semantic/integrity audit;
- [x] residual-MLP seed-1200 smoke test;
- [x] recorded smoke endpoint `component-wise=3072`, `composed=1536`.

This was sufficient for the historical snapshot workflow. It is **not** the stronger current requirement to regenerate the entire `1200–1207` headline cohort under pinned dependencies.

## Version-control boundary — Issue #5

- [x] Create tag `v0.2.0-public-snapshot` at frozen baseline `556dce21c7a5516a16780cb28d528d1ff3968e53`.
- [x] Verify the tag directly resolves to that commit.
- [x] Create the associated GitHub research-snapshot release.
- [x] Complete the independent correction/review sequence before merging PR #7.
- [x] Squash-merge reviewed PR #7 into `main`.
- [x] Delete stale publication-era research branches.
- [x] Confirm the post-merge repository audit passed.

## Historical surviving claim

> Under explicit task distributions, replacement grammars, and passing criteria, some learned spans in the tested networks admit smaller task-preserving replacements when treated as one composed input-output function than when simplified at implementation-component boundaries.

This does **not** justify universal mathematical-complexity, Transformer/LLM, GPU/RAM/energy, or universal runtime claims.

## Why this checklist is no longer the active gate

Since this historical sequence, the repository has continued to be hardened and new candidate external-validity work has been produced on separate research branches. Broad announcement therefore requires another integrated decision over:

- which evidence belongs in the headline claim set;
- whether representative headline results reproduce from a pinned clean environment;
- whether release/citation/README language accurately reflects the final evidence state.

Those decisions are tracked only in `ANNOUNCEMENT_READINESS.md` / Issue #13.

## Preservation rule

Invalidated evidence is retained as provenance and explicitly excluded from inference. The frozen v0.2.0 tag is not rewritten by later corrections or hardening work.