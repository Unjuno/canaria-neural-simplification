# Open research questions

**Project mode: public-snapshot / handoff.**

The purpose of this file is to leave bounded questions that another researcher can pick up. It is not a commitment to continue expanding the current project.

## Completed closure task

### Clean-repository reproduction

A portable public runner now reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` imports. In the recorded environment, the complete output exactly matched the archived confirmatory JSON with SHA256 `68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`.

See `scripts/reproduce/g7_confirmatory/README.md`, `results/reproduction/g7_seed4300_report.json`, and `docs/REPRODUCIBILITY.md`.

This is a software/data portability result for an already-confirmatory seed, not a new independent scientific replication.

## Conditional closure tasks

### 1. Direct compositional-simplification replication on a clearly different family

Only needed if a publication-level novelty/generalization claim requires stronger external validity. Compare component-wise simplification with composed-span simplification under matched replacement/optimization budget, task-utility criterion, complexity accounting, and fresh seeds.

Tracked in GitHub Issue #2.

### 2. Minimal runtime-compilation proof-of-concept

Only needed if systems/deployment claims are made. Measure compact serialized bytes, load/compile/materialization time, peak RAM/VRAM, inference latency/throughput, and task utility/fidelity. A negative result is acceptable.

Tracked in GitHub Issue #3.

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

## Questions that are not open in the tested settings

- Canary is not a necessary local condition for simplification.
- Teacher-forced PPL is not enough to certify autoregressive functional equivalence.
- Merely splitting a direct compiler fit into two stages without task learning does not reproduce the staged benefit.
- Hard shadow-damage vetoes can block successful final contraction.
- The same normalized functional-error threshold is not equally task-safe before and after recontracting.
- A fixed future-risk cap is not enough to produce an automatic cost/utility Pareto improvement.

Before starting new work, read `CORE_DISCOVERY.md`, `CLAIMS_AND_EVIDENCE.md`, `NEGATIVE_RESULTS.md`, `TRAINING_TIME_CONSOLIDATION.md`, and `LATE_STAGE_FINDINGS.md`.
