# Canaria public research snapshot

This file defines how to read the repository as of the current frozen public-snapshot phase.

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
10. `APPLICATIONS.md` — engineering directions and their evidence status.
11. `RUNTIME_POC.md` — bounded CPU-only serialization/materialization/execution proof of concept.
12. `REPRODUCIBILITY.md` — integrity/reproduction policy and portable G7 reproduction.
13. `ROADMAP.md` and `OPEN_QUESTIONS.md` — future research/handoff, not unfinished snapshot work.
14. `HISTORICAL_INDEX.md` — older documents preserved as historical context rather than current instructions.
15. `RELEASE_CHECKLIST.md` — release metadata checklist.
16. `../CHANGELOG.md` — version/snapshot changes.

## Snapshot interpretation

Canaria is not presented as a production-ready compression library. It is an auditable research record centered on the empirical observation that learned computation can sometimes be simpler when represented as a composed task-conditioned function than when treated component-by-component.

Later training-time experiments extend that observation: consolidation followed by task learning changes the ease and task-risk of later consolidation.

The current snapshot is deliberately scoped to the tested settings. It does not require additional experiments to support its present wording.

## Evidence discipline

- Confirmatory, exploratory, reproduction, systems-PoC, and negative evidence remain distinguishable.
- Historical failures are preserved.
- Small-model mechanistic results are not promoted to large-LLM universality claims.
- Parameter reduction, serialized size, compiler-update cost, FLOPs, wall-clock time, memory, and energy are separate quantities.
- A systems PoC result must not be generalized beyond the measured resource and environment.

## Completed portability closure

A self-contained public runner reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` imports.

In the recorded environment, its complete output exactly matched the archived confirmatory JSON with SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

See `scripts/reproduce/g7_confirmatory/` and `results/reproduction/g7_seed4300_report.json`.

This verifies software/data portability for one representative confirmatory pipeline. It is not counted as an additional independent scientific seed.

## Completed bounded runtime PoC

The repository also contains a small CPU-only deployment proof of concept using the same G7 seed-4300 large and progressive compact models.

Measured results:

- serialized artifact + manifest: **110,093 → 54,646 bytes** (`−50.36%`);
- batch-128 CPU inference: **47.05 → 23.11 ms mean** across five fresh-process probes;
- load/materialize: **7.85 → 5.86 ms mean**, secondary because cache sensitivity was observed;
- process RSS delta: **4.72 → 4.56 MB**, so meaningful host-RAM reduction was **not demonstrated**.

The compact artifact executes the learned 2-block representation directly rather than reconstructing the original 4-block model.

See `RUNTIME_POC.md` and `results/reproduction/runtime_poc_seed4300_report.json`.

## Closure state

The current scientific snapshot is **closed at its present claim scope**.

- Issue #1 (portable reproduction) — completed.
- Issue #3 (minimal runtime/materialization PoC) — completed.
- Issue #2 (direct cross-family replication) — closed as `not planned` for v0.2.0 and retained only as a future-work design for a later stronger generalization/priority claim.

No open experiment is required before freezing the current research record.

Future scientific work should begin from a new issue/question or a new research phase rather than silently extending the current G-number sequence.

## Manual release metadata

The remaining optional tasks are repository/UI metadata rather than scientific work:

- set/update the GitHub repository description if desired;
- create tag `v0.2.0-public-snapshot`;
- create release `Canaria v0.2.0 — Public Research Snapshot`.

See `RELEASE_CHECKLIST.md` for suggested text and checks.
