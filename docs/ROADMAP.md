# Research roadmap — public-snapshot / handoff phase

**Current state:** broad experiment expansion is paused.

The earlier cross-architecture roadmap is preserved in `GENERALIZATION_ROADMAP.md` and `GENERALIZATION_STATUS.md` as historical planning/evidence. It should not be read as a current commitment to run every listed experiment.

## What would justify a new experiment now

A new experiment should close one of three concrete gaps:

1. **Public-claim closure** — a direct replication needed to support a claim we intend to make prominently.
2. **Reproducibility closure** — a clean-repository run needed so a third party can reproduce a representative result.
3. **Deployment closure** — a minimal systems proof-of-concept needed before claiming a practical application.

If an experiment does not satisfy one of those criteria, record it under `OPEN_QUESTIONS.md` rather than extending the active mainline.

## Priority 1 — clean-repository reproduction

Goal: from a clean clone and documented dependencies, run at least one representative confirmatory pipeline without private `/mnt/data` assumptions.

Preferred targets:

- one original compositional-simplification result; or
- one training-time staged-vs-direct result.

Success means the public repository contains all required code/data-generation instructions and reproduces the qualitative/registered endpoint. Bitwise reproduction is not required for the oldest historical phases unless the environment is fully specified.

## Priority 2 — direct replication of the core discovery

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

## Priority 3 — minimal runtime-compilation proof-of-concept

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

After the public repository passes its integrity checks and any chosen closure experiment is documented, treat Canaria as a **research snapshot**. New work should start from explicit issues/questions rather than an indefinitely extending G-number sequence.
