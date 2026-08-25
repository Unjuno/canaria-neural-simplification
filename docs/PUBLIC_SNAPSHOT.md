# Canaria public research snapshot

This file defines how to read the repository as of the current consolidation phase.

## Read order

1. `../README.md` — project-level question, main findings, scope.
2. `CORE_DISCOVERY.md` — central empirical discovery: compositional simplification.
3. `CLAIMS_AND_EVIDENCE.md` — current supported/rejected/open claim registry.
4. `TRAINING_TIME_CONSOLIDATION.md` — G7–G17 training-time consolidation evidence.
5. `LATE_STAGE_FINDINGS.md` — G18–G27 mechanism/controller evidence.
6. `NEGATIVE_RESULTS.md` — falsified explanations and failed interventions.
7. `APPLICATIONS.md` — engineering hypotheses and deployment directions.
8. `REPRODUCIBILITY.md` — integrity/reproduction policy.
9. `ROADMAP.md` and `OPEN_QUESTIONS.md` — bounded closure work and handoff.
10. `HISTORICAL_INDEX.md` — which older documents are preserved as historical context rather than current instructions.

## Snapshot interpretation

Canaria is not presented as a production-ready compression library. It is an auditable research record centered on the empirical observation that learned computation can sometimes be simpler when represented as a composed task-conditioned function than when treated component-by-component.

Later training-time experiments extend that observation: consolidation followed by task learning changes the ease and task-risk of later consolidation.

## Evidence discipline

- Confirmatory, exploratory, reproduction, and negative evidence must remain distinguishable.
- Historical failures are preserved.
- Small-model mechanistic results are not promoted to large-LLM universality claims.
- Parameter reduction, serialized size, compiler-update cost, FLOPs, wall-clock time, memory, and energy are separate quantities.
- Runtime-compilation applications remain hypotheses until measured directly.

## Current stopping policy

Broad experiment expansion is paused. New work should close a specific publication, reproduction, or deployment evidence gap. Otherwise it belongs in `OPEN_QUESTIONS.md` for future researchers.

The three concrete closure tasks are tracked as GitHub Issues:

- **#1** — clean-repository reproduction of one confirmatory pipeline;
- **#2** — direct replication of compositional simplification on a different family, only if the public generalization/novelty claim needs it;
- **#3** — minimal runtime-compilation proof of concept.

These issues are intentionally narrow. Completing all three is not required unless the corresponding public claims are pursued.
