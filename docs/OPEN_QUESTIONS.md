# Open research questions

**Project mode: public-snapshot / handoff.**

The purpose of this file is to leave bounded questions that another researcher can pick up. It is not a commitment to continue expanding the current project.

## Completed closure tasks

### Clean-repository reproduction

A portable public runner reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` imports. In the recorded environment, the complete output exactly matched the archived confirmatory JSON with SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

See `scripts/reproduce/g7_confirmatory/README.md`, `results/reproduction/g7_seed4300_report.json`, and `docs/REPRODUCIBILITY.md`.

This is software/data portability evidence for an already-confirmatory seed, not a new independent scientific replication.

### Minimal runtime/materialization proof of concept

A bounded CPU-only PoC now demonstrates:

```text
compact learned representation
→ serialize
→ load/materialize
→ execute directly
```

for G7 seed 4300.

Recorded result:

- serialized artifact + manifest: **110,093 → 54,646 bytes** (`−50.36%`);
- parameters: **23,138 → 11,042** (`−52.28%`);
- batch-128 CPU inference: **47.05 → 23.11 ms mean** over five fresh-process probes;
- load/materialize: **7.85 → 5.86 ms mean**, treated as secondary because cache sensitivity was observed;
- process RSS delta: **4.72 → 4.56 MB**, so meaningful host-RAM reduction was **not demonstrated**;
- test PPL: **19.2784 → 18.9322**.

See `docs/RUNTIME_POC.md` and `results/reproduction/runtime_poc_seed4300_report.json`.

This is a small-model CPU engineering PoC, not evidence of universal GPU, LLM, RAM, or runtime gains.

## Only remaining conditional closure task

### Direct compositional-simplification replication on a clearly different family

This is needed only if a stronger publication-level generalization/novelty claim is pursued.

The decisive design should compare:

- component-wise simplification;
- composed-span simplification;
- matched replacement/optimization budget;
- matched task-utility criterion;
- explicit complexity accounting;
- fresh seeds;
- a clearly different architecture/task family.

The point is to test the **core compositional-simplification phenomenon directly**, not merely to obtain another pruning/compression endpoint.

Tracked in GitHub Issue #2. It is intentionally optional at the current public-claim scope.

## Scientific questions left for future researchers

- Grammar-independent description complexity.
- Large pretrained Transformer/LLM external validity.
- Automatic detection of useful functional boundaries.
- Why recontracting reduces later compiler optimization cost.
- Why downstream task sensitivity can rise at the same time.
- Risk-model transfer across widths, heads, tasks, and architectures.
- Cost-aware autonomous control beyond fixed risk caps.
- Off-manifold versus task-manifold simplification.
- Stronger null models and synthetic teachers with known complexity.
- Stable recursive complexity floors/fixed points.
- Functional IRs and hardware-specific JIT/runtime compilation.
- Whether spanwise materialization can reduce peak RAM/VRAM on realistically large models.

## Questions that are not open in the tested settings

- Canary is not a necessary local condition for simplification.
- Teacher-forced PPL is not enough to certify autoregressive functional equivalence.
- Merely splitting a direct compiler fit into two stages without task learning does not reproduce the staged benefit.
- Hard shadow-damage vetoes can block successful final contraction.
- The same normalized functional-error threshold is not equally task-safe before and after recontracting.
- A fixed future-risk cap is not enough to produce an automatic cost/utility Pareto improvement.
- The current runtime PoC does **not** demonstrate meaningful host-RAM reduction.

Before starting new work, read `CORE_DISCOVERY.md`, `CLAIMS_AND_EVIDENCE.md`, `NEGATIVE_RESULTS.md`, `TRAINING_TIME_CONSOLIDATION.md`, `LATE_STAGE_FINDINGS.md`, and `RUNTIME_POC.md`.
