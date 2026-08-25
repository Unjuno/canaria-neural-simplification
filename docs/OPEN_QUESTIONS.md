# Open research questions

**Project mode: frozen public-snapshot / handoff.**

The purpose of this file is to leave bounded questions that another researcher can pick up. It is not a commitment to continue expanding the current project.

## Completed closure tasks

### Clean-repository reproduction

A portable public runner reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` imports. In the recorded environment, the complete output exactly matched the archived confirmatory JSON with SHA256 `68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`.

See `scripts/reproduce/g7_confirmatory/README.md`, `results/reproduction/g7_seed4300_report.json`, and `REPRODUCIBILITY.md`.

This is a software/data portability result for an already-confirmatory seed, not a new independent scientific replication.

### Minimal runtime/materialization proof of concept

A bounded CPU-only PoC now serializes, materializes, and directly executes the G7 seed-4300 compact learned representation without reconstructing the original 4-block model.

Headline result:

- serialized artifact + manifest: **110,093 → 54,646 bytes** (`−50.36%`);
- batch-128 CPU inference: **47.05 → 23.11 ms mean** across five fresh-process probes;
- meaningful host-RAM reduction was **not demonstrated** (`4.72 → 4.56 MB` RSS delta).

See `RUNTIME_POC.md` and `results/reproduction/runtime_poc_seed4300_report.json`.

## Optional future replication

### Direct compositional-simplification replication on a clearly different family

This is **not required** for the current scoped public claim. It becomes useful only if a future paper/report wants a stronger cross-family generalization or novelty/priority statement.

A future confirmatory design should compare component-wise simplification with composed-span simplification under matched replacement/optimization budget, task-utility criterion, complexity accounting, and fresh seeds.

The previous GitHub Issue #2 records the intended design. For the current snapshot it should be treated as future work, not unfinished closure work.

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
- Functional IRs and hardware-specific JIT/runtime compilation beyond the current small CPU PoC.
- Direct cross-family replication of the core compositional-simplification phenomenon if stronger external-validity language is desired.

## Questions that are not open in the tested settings

- Canary is not a necessary local condition for simplification.
- Teacher-forced PPL is not enough to certify autoregressive functional equivalence.
- Merely splitting a direct compiler fit into two stages without task learning does not reproduce the staged benefit.
- Hard shadow-damage vetoes can block successful final contraction.
- The same normalized functional-error threshold is not equally task-safe before and after recontracting.
- A fixed future-risk cap is not enough to produce an automatic cost/utility Pareto improvement.
- The current runtime PoC does not demonstrate meaningful host-RAM reduction.

Before starting new work, read `CORE_DISCOVERY.md`, `CLAIMS_AND_EVIDENCE.md`, `NEGATIVE_RESULTS.md`, `TRAINING_TIME_CONSOLIDATION.md`, and `LATE_STAGE_FINDINGS.md`.
