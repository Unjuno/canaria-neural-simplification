# Canaria public research snapshot

This file defines how to read the repository as of the current consolidation phase.

## Read order

1. `../README.md` — project-level question, main findings, scope.
2. `CORE_DISCOVERY.md` — central empirical discovery: compositional simplification.
3. `CLAIMS_AND_EVIDENCE.md` — current supported/rejected/open claim registry.
4. `PUBLICATION_NOTES.md` — claim hierarchy for papers, talks, and technical communication.
5. `TRAINING_TIME_CONSOLIDATION.md` — G7–G17 training-time consolidation evidence.
6. `LATE_STAGE_FINDINGS.md` — G18–G27 mechanism/controller evidence.
7. `NEGATIVE_RESULTS.md` — falsified explanations and failed interventions.
8. `TERMINOLOGY.md` — current definitions of Canary, span, compiler, recontracting, fidelity, utility, and complexity terms.
9. `FAQ.md` — common interpretation boundaries and public-facing questions.
10. `APPLICATIONS.md` — engineering hypotheses and deployment directions.
11. `REPRODUCIBILITY.md` — integrity/reproduction policy and the portable G7 reproduction.
12. `ROADMAP.md` and `OPEN_QUESTIONS.md` — bounded conditional work and handoff.
13. `HISTORICAL_INDEX.md` — which older documents are preserved as historical context rather than current instructions.
14. `../CHANGELOG.md` — version/snapshot changes.

## Snapshot interpretation

Canaria is not presented as a production-ready compression library. It is an auditable research record centered on the empirical observation that learned computation can sometimes be simpler when represented as a composed task-conditioned function than when treated component-by-component.

Later training-time experiments extend that observation: consolidation followed by task learning changes the ease and task-risk of later consolidation.

## Evidence discipline

- Confirmatory, exploratory, reproduction, and negative evidence must remain distinguishable.
- Historical failures are preserved.
- Small-model mechanistic results are not promoted to large-LLM universality claims.
- Parameter reduction, serialized size, compiler-update cost, FLOPs, wall-clock time, memory, and energy are separate quantities.
- Runtime-compilation applications remain hypotheses until measured directly.

## Completed portability closure

A self-contained public runner now reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` imports.

In the recorded environment, its complete output exactly matched the archived confirmatory JSON with SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

See `scripts/reproduce/g7_confirmatory/` and `results/reproduction/g7_seed4300_report.json`.

This verifies software/data portability for one representative confirmatory pipeline. It is not counted as an additional independent scientific seed.

## Current stopping policy

Broad experiment expansion is paused. The repository now satisfies the representative clean-reproduction closure target.

Only two conditional public-claim tasks remain:

- **Issue #2** — direct replication of compositional simplification on a different family, only if a stronger public generalization/novelty claim is pursued;
- **Issue #3** — minimal runtime-compilation proof of concept, only if deployment/runtime claims are pursued.

Issue #1, clean-repository reproduction, is complete and should remain closed unless a portability regression is found.

Neither remaining issue is required to preserve the current research snapshot at its present claim scope.
