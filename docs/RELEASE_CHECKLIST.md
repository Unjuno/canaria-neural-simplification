# Public snapshot release checklist

This checklist records the post-independent-re-review repository state. It separates the scientific quality gate (Issue #9) from the later release/tag/merge gate (Issue #5).

## Scientific/public-claim review

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

## Repository quality gate

The reviewed head immediately before this checklist-only update (`02dc5ed3cecb9c767520b60b58dc1b226c1d4b9a`) passed GitHub `repository-audit`, including:

- [x] reusable codec unit tests;
- [x] `python tools/audit_repo.py` semantic/integrity audit;
- [x] minimal public residual-MLP runner smoke test for seed 1200;
- [x] recorded smoke endpoint `component-wise=3072`, `composed=1536`;
- [x] root `README.md` / `STATUS.md` updated to the re-reviewed publication-gate state.

Because this file update creates a new head commit, **Issue #9 is closed only after `repository-audit` also passes on this checklist-only final head**. No further scientific or documentation edits should be made between that final PASS and Issue #9 closure.

## Separate release/merge boundary — Issue #5

Closing Issue #9 completes the independent scientific quality gate. It does **not** authorize public posting by itself.

After #9 closes, follow Issue #5:

1. Create tag `v0.2.0-public-snapshot` at frozen baseline `556dce21c7a5516a16780cb28d528d1ff3968e53`.
2. Create release `Canaria v0.2.0 — Public Research Snapshot` from that tag.
3. Re-review PR #7 against the version boundary and **squash-merge** it if its final audit passes.
4. Delete merged/stale research branches as specified by Issue #5.
5. Confirm final `main` CI/audit passes.
6. Only then post/share the repository.

## Public claim that survives the gate

> Under explicit task distributions, replacement grammars, and passing criteria, some learned spans in the tested networks admit smaller task-preserving replacements when treated as one composed input-output function than when simplified at implementation-component boundaries.

This does **not** justify universal mathematical-complexity, Transformer/LLM, GPU/RAM/energy, or universal runtime claims.

## Citation

Until a paper exists, cite the exact commit/tag and the protocol/result artifacts supporting the specific endpoint. Invalidated artifacts may be cited as correction history, never as scientific support.
