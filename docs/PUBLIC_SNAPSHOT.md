# Canaria public research snapshot

This file defines how to read the repository as of the current frozen public-snapshot phase.

## Read order

1. `../README.md` — project-level question, main findings, scope.
2. `CORE_DISCOVERY.md` — central empirical discovery: compositional simplification.
3. `CROSS_FAMILY_COMPOSITION_REPLICATION.md` — fresh SmallViT component-wise versus composed replication.
4. `CORE_DISCOVERY_REPLICATION_DIGITS.md` — fresh residual-MLP exact-budget replication and joint span-objective control.
5. `CLAIMS_AND_EVIDENCE.md` — current supported/rejected/open claim registry.
6. `PUBLICATION_NOTES.md` — claim hierarchy for papers, talks, and technical communication.
7. `TRAINING_TIME_CONSOLIDATION.md` — G7–G17 training-time consolidation evidence.
8. `LATE_STAGE_FINDINGS.md` — G18–G27 mechanism/controller evidence.
9. `NEGATIVE_RESULTS.md` — falsified explanations and failed interventions.
10. `TERMINOLOGY.md` — current definitions of Canary, span, compiler, recontracting, fidelity, utility, and complexity terms.
11. `FAQ.md` — common interpretation boundaries and public-facing questions.
12. `APPLICATIONS.md` — engineering directions and their evidence status.
13. `RUNTIME_POC.md` — bounded CPU-only serialization/materialization/execution proof of concept.
14. `REPRODUCIBILITY.md` — integrity/reproduction policy and portable G7 reproduction.
15. `ROADMAP.md` and `OPEN_QUESTIONS.md` — future research/handoff, not unfinished snapshot work.
16. `HISTORICAL_INDEX.md` — older documents preserved as historical context rather than current instructions.
17. `RELEASE_CHECKLIST.md` — release metadata checklist.
18. `../CHANGELOG.md` — version/snapshot changes.

## Snapshot interpretation

Canaria is not presented as a production-ready compression library. It is an auditable research record centered on the empirical observation that learned computation can sometimes be simpler when represented as a composed task-conditioned function than when treated component-by-component.

The static core phenomenon now has:

- original residual-CNN confirmatory composition evidence;
- a fresh direct SmallViT replication under a locked component-wise-versus-composed comparison; and
- a second fresh residual-MLP replication under exact learned-parameter-budget matching.

Later training-time experiments extend that observation: consolidation followed by task learning changes the ease and task-risk of later consolidation.

The current snapshot is deliberately scoped to the tested settings. No additional experiment is required to support its present wording.

## Evidence discipline

- Confirmatory, exploratory, reproduction, systems-PoC, and negative evidence remain distinguishable.
- Historical failures are preserved.
- Small-model mechanistic results are not promoted to large-LLM universality claims.
- Parameter reduction, serialized size, compiler-update cost, FLOPs, wall-clock time, memory, and energy are separate quantities.
- A systems PoC result must not be generalized beyond the measured resource and environment.
- The SmallViT and residual-MLP replications strengthen architecture-family external validity but do not establish task-universal, grammar-independent, Transformer-universal, or LLM-universal subadditivity.

## Completed direct architecture-family replications

### Small Vision Transformer

Under the locked passing rule (`held-out span NMSE <= 0.12`, validation utility `>= 0.95`) across the first 8 fresh baseline-eligible seeds `>=9000`:

- component-wise minimum passing complexity: **9,808 replacement params** in all 8 seeds;
- composed minimum passing complexity: **4,904–5,424 params**;
- mean composed/component-wise complexity ratio: **0.51988**;
- seed-bootstrap95: **[0.50634, 0.53926]**;
- composed smaller: **8/8**;
- selected composed mean test utility: **0.97856**, test data not used for selection.

See `CROSS_FAMILY_COMPOSITION_REPLICATION.md` and `results/replication/vit_compositional/`.

### Residual MLP

A four-block residual MLP on sklearn digits used a fixed first-two-block span. At each budget point the learned replacement-parameter count was exactly matched between component-wise and composed conditions.

Fresh seeds `1200–1207`:

- component-wise mean minimum passing budget: **3584 params**;
- composed mean minimum passing budget: **1728 params**;
- composed smaller: **8/8**;
- mean `log2(B_composed/B_componentwise)`: **−1.0519**;
- seed-bootstrap95: **[−1.2075, −0.8962]**;
- geometric mean budget ratio: **0.4823×**;
- untouched-test accuracy difference at validation-selected budgets: composed minus component-wise **+0.583 pt**, bootstrap95 **[+0.306,+0.806] pt**.

At fixed 2048 params, a mechanistic control preserved the two-module topology but changed the objective from local intermediate targets to the composed span target:

- local component-wise NMSE: **0.1474**;
- jointly fit two-module span: **0.0639**;
- one composed module: **0.0533**.

Most of the gap therefore follows the **functional span objective/boundary**, not merely the topology change.

See `CORE_DISCOVERY_REPLICATION_DIGITS.md` and `results/core_discovery_digits/`.

## Completed portability closure

A self-contained public runner reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` imports.

In the recorded environment, its complete output exactly matched the archived confirmatory JSON with SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

See `scripts/reproduce/g7_confirmatory/` and `results/reproduction/g7_seed4300_report.json`.

## Completed bounded runtime PoC

The repository also contains a small CPU-only deployment proof of concept using the same G7 seed-4300 large and progressive compact models.

Measured results:

- serialized artifact + manifest: **110,093 → 54,646 bytes** (`−50.36%`);
- batch-128 CPU inference: **47.05 → 23.11 ms mean**;
- load/materialize: **7.85 → 5.86 ms mean**, secondary because cache sensitivity was observed;
- process RSS delta: **4.72 → 4.56 MB**, so meaningful host-RAM reduction was **not demonstrated**.

See `RUNTIME_POC.md` and `results/reproduction/runtime_poc_seed4300_report.json`.

## Closure state

The current scientific snapshot is **closed at its present claim scope**.

- Issue #1 — portable reproduction: completed.
- Issue #2 — direct cross-family compositional replication: completed; later strengthened by the additional residual-MLP replication.
- Issue #3 — minimal runtime/materialization PoC: completed.

No open experiment is required before freezing the current research record.

Future scientific work should begin from a new issue/question or a new research phase rather than silently extending the current G-number sequence.

## Manual release metadata

The remaining optional tasks are repository/UI metadata rather than scientific work:

- set/update the GitHub repository description if desired;
- create tag `v0.2.0-public-snapshot`;
- create release `Canaria v0.2.0 — Public Research Snapshot`.

See `RELEASE_CHECKLIST.md` for suggested text and checks.
