# Research roadmap — publication gate / future-work handoff

**Current state:** broad experiment expansion is stopped. The 2026-08-26 independent scientific re-review has been performed; Issue #9 remains the quality gate until final smoke/audit checks pass and the issue closes. Issue #5 then governs the separate release/tag/merge sequence.

## Completed scientific/reproducibility work retained

### Portable G7 reproduction

A public runner reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` assumptions and exactly matched the archived output in its recorded environment. This is reproduction/portability evidence, not a new independent scientific replication.

### Bounded runtime PoC

For G7 seed 4300, the small CPU PoC reported a smaller serialized artifact and lower measured batch-128 CPU inference latency. Meaningful host-RAM reduction was not demonstrated; general GPU/VRAM/energy/LLM runtime benefits remain open.

### Direct compositional replications

**Residual MLP:** fresh `1200–1207`, exact learned replacement-parameter matching at every grid point, composed lower minimum passing budget in `8/8`, geometric ratio `0.4823×`. Validation selects the endpoint; test follows selection.

**SmallViT:** fresh locked protocol, composed selected replacement smaller in `8/8`, mean ratio `0.51988`. Re-review caveat: the selection rule excludes test accuracy, but the runner records test metrics for all candidates, so test was not operationally hidden during candidate-result generation.

The residual-MLP 2048-parameter joint-factorized result is descriptive/mechanistic secondary and is consistent with an important role for the composed span objective; it is not a confirmatory causal decomposition.

## Phase 2 correction that must remain visible

Phase 2E is `INVALIDATED_IMPLEMENTATION_BUG` and `DO_NOT_USE_FOR_INFERENCE`. It used raw `Xt` where the replacement was defined on internal activation `ta[0]`; equal width 64 hid the semantic error.

The invalid result remains provenance, not negative scientific evidence. Phase 2I's RNG explanation is retracted, and Phase 2O did not confirm a reliable composed repair-sample advantage.

## Current stopping rule

Until Issue #9 closes, do only:

- public claim/provenance corrections;
- minimal public-runner smoke checks;
- CI/audit fixes;
- version/release/PR preparation.

After Issue #9, follow Issue #5 for:

1. v0.2.0 baseline tag/release;
2. PR #7 final review and squash-merge if appropriate;
3. stale branch cleanup;
4. final `main` CI/audit;
5. public posting/sharing.

Do not reopen broad experiments merely to complete publication.

## Future research topics

Scientifically interesting but separate from this gate:

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
