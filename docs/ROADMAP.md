# Research roadmap — public-snapshot / handoff phase

**Current state:** broad experiment expansion is paused.

The earlier cross-architecture roadmap is preserved in `GENERALIZATION_ROADMAP.md` and `GENERALIZATION_STATUS.md` as historical planning/evidence. It should not be read as a current commitment to run every listed experiment.

## What would justify a new experiment now

A new experiment should close one of two remaining conditional gaps:

1. **Public-claim closure** — a direct replication needed to support a stronger generalization/novelty claim we intend to make prominently.
2. **Deployment closure** — a minimal systems proof-of-concept needed before claiming a practical runtime/application benefit.

If an experiment does not satisfy one of those criteria, record it under `OPEN_QUESTIONS.md` rather than extending the active mainline.

## Completed closure — clean-repository reproduction

A portable runner now reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` assumptions.

The current public reproduction uses only scikit-learn's packaged dataset-description text as source material and preserves the historical model, data-window generation, seeds, minibatch order, optimizer schedule, compiler budgets, and metrics.

In the recorded environment, the generated JSON exactly matched the archived confirmatory seed output, including SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

Public artifacts:

- `scripts/reproduce/g7_confirmatory/run_seed.py`
- `scripts/reproduce/g7_confirmatory/README.md`
- `results/reproduction/g7_seed4300_report.json`
- `.github/workflows/reproduce-g7.yml`

This closes the repository-portability gap for one representative confirmatory pipeline. It does not constitute a new independent scientific replication because seed 4300 belongs to the original G7 confirmatory cohort.

## Conditional priority 1 — direct replication of the core discovery

Only if a publication-level novelty/generalization claim requires it, run one clearly different architecture/task that directly tests:

> component-wise simplification versus composed-span simplification under matched fidelity/utility and complexity accounting.

The point is to replicate **compositional simplification itself**, not merely another pruning/compression endpoint.

A useful confirmatory design should predefine:

- component and composed spans;
- replacement grammar/budget;
- task-utility criterion;
- complexity measure(s);
- fresh seed policy;
- paired decision rule.

Tracked in GitHub Issue #2.

## Conditional priority 2 — minimal runtime-compilation proof-of-concept

Only if deployment claims are to be made, build one small end-to-end demonstration:

```text
compact functional representation
→ load
→ materialize/compile
→ execute
```

Measure at minimum:

- serialized bytes;
- compile/materialization latency;
- peak host/device memory if measurable;
- inference latency;
- task utility/fidelity.

A negative result is acceptable. The goal is to separate storage/distribution benefit from runtime-memory or execution-speed benefit.

Tracked in GitHub Issue #3.

## Handoff topics for future researchers

These remain interesting but are not required to close the current project:

- large pretrained Transformer/LLM external validity;
- codec-independent complexity/MDL;
- off-manifold functional complexity;
- stronger null models and known-complexity synthetic teachers;
- effective repair/tangent dimension;
- mechanism algebra/dictionaries;
- sensitivity-aware utility-cost controllers;
- hardware-specific functional IR and JIT execution.

See `OPEN_QUESTIONS.md` for the bounded handoff list.

## Stopping rule

The repository now has a public-snapshot reading order, automated integrity checks, negative-result preservation, and one portable exact confirmatory reproduction. Treat Canaria as a **research snapshot** unless one of the two conditional public-claim tasks above is deliberately pursued.

New work should start from explicit issues/questions rather than an indefinitely extending G-number sequence.
