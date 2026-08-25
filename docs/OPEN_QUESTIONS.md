# Open research questions

**Project mode: frozen public-snapshot / handoff.**

The purpose of this file is to leave bounded questions that another researcher can pick up. It is not a commitment to continue expanding the current project.

## Completed closure tasks

### Clean-repository reproduction

A portable public runner reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` imports. In the recorded environment, the complete output exactly matched the archived confirmatory JSON with SHA256 `68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`.

### Minimal runtime/materialization proof of concept

A bounded CPU-only PoC serializes, materializes, and directly executes the G7 seed-4300 compact learned representation without reconstructing the original 4-block model.

Headline result:

- serialized artifact + manifest: **110,093 → 54,646 bytes** (`−50.36%`);
- batch-128 CPU inference: **47.05 → 23.11 ms mean**;
- meaningful host-RAM reduction was **not demonstrated** (`4.72 → 4.56 MB` RSS delta).

### Direct cross-family replication of compositional simplification

The core phenomenon now has two direct fresh architecture-family replications beyond the original residual-CNN setting.

**SmallViT:**

- component-wise minimum passing complexity: **9,808 replacement params**;
- composed minimum passing complexity: **4,904–5,424 params**;
- mean complexity ratio: **0.51988**;
- paired bootstrap95: **[0.50634, 0.53926]**;
- composed smaller: **8/8**;
- selected composed mean test utility: **0.97856**.

**Residual MLP:**

- component-wise mean minimum passing budget: **3584 params**;
- composed mean minimum passing budget: **1728 params**;
- composed lower-budget: **8/8 fresh seeds**;
- mean `log2(B_composed/B_componentwise) = -1.0519`, bootstrap95 **[-1.2075,-0.8962]**;
- untouched-test accuracy difference at validation-selected budgets: **+0.583 pt**, bootstrap95 **[+0.306,+0.806] pt**.

At fixed 2048 params, local component-wise NMSE was **0.1474**, the same two-module architecture jointly fit to the composed span target reached **0.0639**, and one composed module reached **0.0533**. This strengthens the functional-boundary/objective interpretation.

See `CROSS_FAMILY_COMPOSITION_REPLICATION.md` and `CORE_DISCOVERY_REPLICATION_DIGITS.md`.

## Scientific questions left for future researchers

- Grammar-independent description complexity.
- Larger pretrained Transformer/LLM external validity.
- Replication across genuinely different **task types** rather than only additional architecture families.
- Replication across additional spans, widths, and replacement grammars.
- Automatic detection of useful functional boundaries.
- Why recontracting reduces later compiler optimization cost.
- Why downstream task sensitivity can rise at the same time.
- Risk-model transfer across widths, heads, tasks, and architectures.
- Cost-aware autonomous control beyond fixed risk caps.
- Off-manifold versus task-manifold simplification.
- Stronger null models and synthetic teachers with known complexity.
- Stable recursive complexity floors/fixed points.
- Functional IRs and hardware-specific JIT/runtime compilation beyond the current small CPU PoC.

## Questions that are not open in the tested settings

- Canary is not a necessary local condition for simplification.
- Teacher-forced PPL is not enough to certify autoregressive functional equivalence.
- Merely splitting a direct compiler fit into two stages without task learning does not reproduce the staged benefit.
- Hard shadow-damage vetoes can block successful final contraction.
- The same normalized functional-error threshold is not equally task-safe before and after recontracting.
- A fixed future-risk cap is not enough to produce an automatic cost/utility Pareto improvement.
- The current runtime PoC does not demonstrate meaningful host-RAM reduction.
- The SmallViT and residual-MLP replications do not establish universal Transformer, LLM, task-universal, or grammar-independent compositional subadditivity.

Before starting new work, read `CORE_DISCOVERY.md`, `CROSS_FAMILY_COMPOSITION_REPLICATION.md`, `CORE_DISCOVERY_REPLICATION_DIGITS.md`, `CLAIMS_AND_EVIDENCE.md`, `NEGATIVE_RESULTS.md`, `TRAINING_TIME_CONSOLIDATION.md`, and `LATE_STAGE_FINDINGS.md`.
