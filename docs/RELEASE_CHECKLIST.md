# Public snapshot release checklist

This checklist reflects the **post-independent-re-review** repository state. It must not be used to imply publication readiness before the remaining gates are actually satisfied.

## Scientific/public-claim review

- [x] Root README centered on the bounded compositional-simplification claim.
- [x] Public claim registry separates operational claims from mathematical/Kolmogorov claims.
- [x] Residual-MLP direct replication re-reviewed for split isolation, exact learned-budget accounting, seed rules, and statistics.
- [x] SmallViT direct replication re-reviewed; test metrics are excluded from the locked selection rule, but the runner's all-candidate test recording is disclosed as a weaker isolation boundary.
- [x] Training-time primary/secondary and exploratory/confirmatory boundaries re-reviewed.
- [x] Phase 2E marked `INVALIDATED_IMPLEMENTATION_BUG`; causal descendants corrected/retracted.
- [x] Phase 2O positive repair-sample-complexity claim removed; result remains `UNCERTAIN`.
- [x] Invalidated evidence preserved as correction history rather than silently erased.
- [x] G7 exact rerun described as reproduction/portability evidence, not an independent scientific replication.
- [x] Runtime PoC restricted to the measured small CPU/storage/inference scope.
- [x] Independent re-review ledger added: `INDEPENDENT_REREVIEW_2026-08-26.md`.

## Repository quality gate

Before closing Issue #9:

- [ ] Minimal public residual-MLP runner smoke test passes on the reviewed branch.
- [ ] `python tools/audit_repo.py` passes with Phase 2 invalidation invariants enabled.
- [ ] GitHub `repository-audit` workflow passes for the final reviewed commit.
- [ ] Root `README.md` / `STATUS.md` updated from “review pending” to the actual completed-review state.
- [ ] Issue #9 receives the final decision summary and is closed.

## Separate release/merge boundary

Closing Issue #9 means the independent scientific quality gate is complete. It does **not** by itself authorize merging PR #7 or rewriting the frozen v0.2.0 boundary.

Before merging or posting the post-snapshot branch:

1. Resolve the v0.2.0 tag/release boundary tracked separately from Issue #9.
2. Re-review draft PR #7 against that version boundary.
3. Confirm the final PR head still passes `repository-audit`.
4. Keep Phase 2 post-snapshot claims version-separated from the frozen v0.2.0 claim set unless an explicit version transition is made.

## Public claim that survives the gate

The scoped statement is:

> Under explicit task distributions, replacement grammars, and passing criteria, some learned spans in the tested networks admit smaller task-preserving replacements when treated as one composed input-output function than when simplified at implementation-component boundaries.

This does **not** justify universal mathematical-complexity, LLM-scale, GPU/RAM/energy, or universal runtime claims.

## Citation

Until a paper exists, cite the exact commit/tag and the protocol/result artifacts supporting the particular claim. Do not cite an invalidated artifact as evidence merely because it remains in history.
