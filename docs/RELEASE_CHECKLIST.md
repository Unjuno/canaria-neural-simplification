# Public snapshot release checklist

This checklist records the completed independent-review and publication sequence. It separates the frozen v0.2.0 release boundary from the reviewed post-snapshot state on `main`.

## Scientific/public-claim review — Issue #9

- [x] Root README centered on the bounded operational compositional-simplification claim.
- [x] Public claim registry separates replacement/description complexity from mathematical/Kolmogorov complexity.
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

## Repository quality gate

The independently reviewed tree passed GitHub `repository-audit`, including:

- [x] reusable codec unit tests;
- [x] `python tools/audit_repo.py` semantic/integrity audit;
- [x] minimal public residual-MLP runner smoke test for seed 1200;
- [x] recorded smoke endpoint `component-wise=3072`, `composed=1536`.

The release-boundary helper workflow was temporary. After it was deleted, the PR head had **zero file differences** from the independently reviewed head before squash merge.

## Release/version-control boundary — Issue #5

- [x] Create tag `v0.2.0-public-snapshot` at frozen baseline `556dce21c7a5516a16780cb28d528d1ff3968e53`.
- [x] Verify the tag directly resolves to that commit.
- [x] Create release `Canaria v0.2.0 — Public Research Snapshot` from that tag.
- [x] Mark PR #7 ready after Issue #9 closure.
- [x] Confirm the restored PR tree matches the independently reviewed tree.
- [x] Confirm PR-head `repository-audit` PASS.
- [x] Squash-merge PR #7 into `main`.
- [x] Delete `research/phase2-precision-quantization` and `research-snapshot-2026-08-24`.
- [x] Confirm `main` is the sole active branch.
- [x] Confirm final current-`main` `repository-audit` PASS; the immutable run/result is recorded in Issue #5 rather than by making another self-invalidating checklist commit.

## Surviving public claim

> Under explicit task distributions, replacement grammars, and passing criteria, some learned spans in the tested networks admit smaller task-preserving replacements when treated as one composed input-output function than when simplified at implementation-component boundaries.

This does **not** justify universal mathematical-complexity, Transformer/LLM, GPU/RAM/energy, or universal runtime claims.

## Preservation rule

Invalidated evidence is retained as provenance and explicitly excluded from inference. The frozen v0.2.0 tag is not rewritten by post-snapshot corrections.

## Citation

Until a paper exists, cite the exact commit/tag and the protocol/result artifacts supporting the specific endpoint. Invalidated artifacts may be cited as correction history, never as scientific support.
