# Research roadmap — published snapshot / future-work handoff

**Current state:** the publication-quality gate is complete. Broad experiment expansion is stopped. The frozen v0.2.0 baseline is tagged/released, the independently reviewed post-snapshot state is on `main`, and stale research branches have been removed.

## Completed closure — scientific/public review

The 2026-08-26 independent re-review classified material public claims as `KEEP`, `EDIT`, `REMOVE`, or `INVALIDATE`.

Key corrections retained in the public baseline:

- operational replacement/description complexity is kept distinct from mathematical/Kolmogorov complexity;
- residual-MLP exact-budget replication retained;
- SmallViT retained with its test-recording isolation caveat;
- G7/G18 primary-versus-secondary wording narrowed;
- Phase 2E preserved as `INVALIDATED_IMPLEMENTATION_BUG`, not negative evidence;
- Phase 2I RNG causal claim retracted;
- Phase 2O remains `UNCERTAIN`, so no reliable compositional repair-sample advantage is claimed.

See `INDEPENDENT_REREVIEW_2026-08-26.md`.

## Completed closure — clean-repository reproduction

A public runner reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` assumptions and exactly matched the archived output in its recorded environment. This is reproduction/portability evidence, not a new independent scientific replication.

## Completed closure — bounded runtime PoC

For G7 seed 4300, the small CPU PoC reported a smaller serialized artifact and lower measured batch-128 CPU inference latency. Meaningful host-RAM reduction was not demonstrated; general GPU/VRAM/energy/LLM runtime benefits remain open.

## Completed closure — direct compositional replications

**Residual MLP:** fresh `1200–1207`, exact learned replacement-parameter matching at every grid point, composed lower minimum passing budget in `8/8`, geometric ratio `0.4823×`. Validation selects the endpoint; test follows selection.

**SmallViT:** fresh locked protocol, composed selected replacement smaller in `8/8`, mean ratio `0.51988`. Re-review caveat: the selection rule excludes test accuracy, but the runner records test metrics for all candidates, so test was not operationally hidden during candidate-result generation.

The residual-MLP 2048-parameter joint-factorized result is descriptive/mechanistic secondary and is consistent with an important role for the composed span objective; it is not a confirmatory causal decomposition.

## Completed closure — release/version-control boundary

- `v0.2.0-public-snapshot` preserves commit `556dce21c7a5516a16780cb28d528d1ff3968e53`.
- GitHub release: `Canaria v0.2.0 — Public Research Snapshot`.
- PR #7 was squash-merged after the frozen release boundary was established.
- stale `research/phase2-precision-quantization` and `research-snapshot-2026-08-24` branches were deleted.
- `main` is the sole active branch.

## Current stopping rule

Do not continue the old publication/closure sequence or add broad experiments merely to strengthen the public presentation.

New scientific work should start from a new issue/research phase with:

1. a falsifiable claim;
2. explicit evidence class and protocol lock where appropriate;
3. an independently initialized inferential unit/seed policy;
4. a stopping rule;
5. explicit separation from the frozen v0.2.0 tag and the reviewed post-snapshot baseline.

## Future research topics

Scientifically interesting but separate from the completed publication gate:

- larger pretrained Transformer/LLM external validity;
- different task types and spans;
- additional replacement grammars and stronger nulls;
- codec-independent complexity / MDL;
- off-manifold functional complexity;
- effective repair/tangent dimension;
- mechanism dictionaries/algebra;
- sensitivity-aware utility-cost controllers;
- hardware-specific functional IR/JIT;
- larger-scale RAM/VRAM/energy/runtime benchmarking.

See `OPEN_QUESTIONS.md` for future-work handoff.
