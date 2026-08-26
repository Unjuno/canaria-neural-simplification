# Open research questions

**Project mode: publication-quality gate / handoff.**

This file lists future questions. It is not a reason to open broad new experiments before Issue #9 and Issue #5 are completed.

## Evidence already retained

### Public reproduction / runtime

- G7 seed 4300 has an exact recorded-environment public reproduction; this is software/portability evidence, not independent replication.
- One small CPU PoC showed a smaller serialized artifact and lower measured batch-128 CPU inference latency; meaningful host-RAM reduction was not demonstrated.

### Direct compositional replications

**Residual MLP — strongest matched-budget public comparison:**

- fresh `1200–1207`;
- exact learned replacement-parameter matching at every grid point;
- component-wise mean minimum passing budget `3584`;
- composed `1728`;
- composed lower `8/8`;
- geometric ratio `0.4823×`;
- validation chooses the endpoint; test follows selection.

At fixed 2048 parameters, the same two-module topology fitted jointly to the span target recovered most of the local component-wise NMSE gap. This is **descriptive/mechanistic secondary** and is consistent with an important role for the composed span objective; it is not a confirmatory causal decomposition.

**SmallViT:**

- composed selected replacement smaller in `8/8` fresh eligible seeds;
- mean replacement-parameter ratio `0.51988`;
- bootstrap95 `[0.50634,0.53926]`.

Re-review caveat: the locked selection rule excludes test accuracy, but the runner records test metrics for all candidates. Test was not a selection variable, yet it was not operationally hidden during candidate-result generation.

### Phase 2 correction

Phase 2E is `INVALIDATED_IMPLEMENTATION_BUG`, not valid negative evidence. Its raw-`Xt` versus internal-activation-domain error invalidates the `0/8` result for inference. Phase 2I's RNG explanation is retracted. Phase 2O did not confirm a reliable composed repair-sample advantage.

## Scientific questions left for future work

- Grammar-independent description complexity / MDL.
- Larger pretrained Transformer/LLM external validity.
- Replication across genuinely different task types, spans, widths, and replacement grammars.
- Stronger SmallViT-style replication with operationally hidden test evaluation.
- Automatic detection of useful functional boundaries.
- Which mechanisms explain the gap between local component targets and composed span objectives?
- Why recontracting can reduce later compiler optimization cost while increasing downstream task sensitivity.
- Risk-model transfer across architectures/tasks.
- Cost-aware autonomous control beyond fixed risk caps.
- Off-manifold versus task-manifold simplification.
- Stronger null models and synthetic teachers with known complexity.
- Stable recursive complexity floors/fixed points.
- Whether a compositional QAT repair-sample advantage exists under stronger preregistered designs; current Phase 2O is uncertain.
- Functional IRs and hardware-specific JIT/runtime compilation beyond the current small CPU PoC.

## Questions already constrained in the tested settings

- Canary is not a necessary local condition.
- Teacher-forced PPL does not certify autoregressive trajectory equivalence.
- Back-to-back factorized compiler fitting without intervening task learning does not reproduce the tested staged benefit.
- Hard shadow-damage vetoes can block final contraction.
- Matched normalized functional error is not matched task safety before versus after recontracting.
- A fixed future-risk cap did not establish an automatic cost/utility Pareto improvement.
- The current runtime PoC does not demonstrate meaningful host-RAM reduction.
- The SmallViT/residual-MLP results do not establish universal mathematical, Transformer, LLM, task-universal, or grammar-independent subadditivity.
- The bugged Phase 2E result cannot be used as a negative result.

Before starting new work, read `INDEPENDENT_REREVIEW_2026-08-26.md`, `CLAIMS_AND_EVIDENCE.md`, `CORE_DISCOVERY.md`, `phase2/README.md`, and `NEGATIVE_RESULTS.md`.
