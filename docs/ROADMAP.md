# Research roadmap — public-snapshot / handoff phase

**Current state:** broad experiment expansion is paused.

The earlier cross-architecture roadmap is preserved in `GENERALIZATION_ROADMAP.md` and `GENERALIZATION_STATUS.md` as historical planning/evidence. It should not be read as a current commitment to run every listed experiment.

## What would justify a new experiment now

Only one remaining conditional experiment is considered potentially useful for a stronger public claim:

- **direct replication of compositional simplification on a clearly different architecture/task**, if a stronger generalization/novelty claim is deliberately pursued.

Everything else belongs in `OPEN_QUESTIONS.md` as future work rather than an active mainline obligation.

## Completed closure — clean-repository reproduction

A portable runner reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` assumptions.

The generated JSON exactly matched the archived confirmatory seed output in the recorded environment, including SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

Public artifacts:

- `scripts/reproduce/g7_confirmatory/run_seed.py`
- `scripts/reproduce/g7_confirmatory/README.md`
- `results/reproduction/g7_seed4300_report.json`
- `.github/workflows/reproduce-g7.yml`

This closes the repository-portability gap for one representative confirmatory pipeline. It is not a new independent scientific replication.

## Completed closure — minimal runtime materialization PoC

A bounded CPU-only systems PoC now implements:

```text
trained large / compact model
→ serialize state + manifest
→ fresh-process load/materialize
→ direct execution
```

For G7 seed 4300:

- serialized artifact + manifest: **110,093 → 54,646 bytes** (**−50.36%**);
- parameters: **23,138 → 11,042** (**−52.28%**);
- mean CPU batch-128 inference across five fresh processes: **47.05 → 23.11 ms** (compact/large **0.491×**);
- mean load/materialize: **7.85 → 5.86 ms**, but cache sensitivity makes this secondary;
- RSS delta: **4.72 → 4.56 MB** (compact/large **0.966×**), so meaningful host-RAM reduction was **not demonstrated**.

The compact artifact executes the learned 2-block compiler directly rather than rebuilding the original 4-block model.

Public artifacts:

- `docs/RUNTIME_POC.md`
- `scripts/reproduce/g7_confirmatory/runtime_poc.py`
- `results/reproduction/runtime_poc_seed4300_report.json`
- `.github/workflows/runtime-poc.yml`

This closes the minimal deployment-PoC task at a small-model CPU scope. It does not establish GPU, LLM, or universal runtime benefits.

## Conditional priority — direct replication of the core discovery

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

## Handoff topics for future researchers

These remain interesting but are not required to close the current project:

- large pretrained Transformer/LLM external validity;
- codec-independent complexity/MDL;
- off-manifold functional complexity;
- stronger null models and known-complexity synthetic teachers;
- effective repair/tangent dimension;
- mechanism algebra/dictionaries;
- sensitivity-aware utility-cost controllers;
- hardware-specific functional IR and JIT execution beyond the minimal PoC.

See `OPEN_QUESTIONS.md` for the bounded handoff list.

## Stopping rule

The repository now has a public-snapshot reading order, automated integrity checks, negative-result preservation, one portable exact confirmatory reproduction, and one reproducible small-model runtime materialization PoC.

Treat Canaria as a **research snapshot** at its current claim scope. New scientific work should start from explicit issues/questions rather than an indefinitely extending G-number sequence.
