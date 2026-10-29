# Canaria

**Task-conditioned neural simplification: a limited research report, with code, evidence, and failures.**

[日本語](README.ja.md) · [Research assessment](RESEARCH_CHECKPOINT.md) · [Status](STATUS.md) · [Historical studies](docs/RESEARCH_INDEX.md)

Can a trained neural-network span be replaced more economically by approximating its composed input–output function rather than its implementation blocks separately?

## Current conclusion

**There is a conditional benefit in a small, fixed Digits experiment—not an established general or production compression method.** The replacement span becomes smaller on average, but the whole model barely shrinks. The examined CIFAR-10 series did not meet the required quality.

Start with the **[current research assessment](RESEARCH_CHECKPOINT.md)** for what worked, what failed, what remains unknown, and whether to continue. The **[full report](research-checkpoints/2026-10-27/README.md)** is the canonical source for methods, confidence intervals, evidence, and reproduction boundaries. This page summarizes existing results; it introduces no new experiment.

## What the recorded results show

In Digits, the first two blocks of a four-block residual MLP were replaced; the remaining network stayed fixed. **J** jointly trains two maps; **C** approximates the same endpoint with one map. Sixteen teachers share one dataset split—not sixteen independent datasets. Passing requires validation endpoint NMSE ≤ 0.08 and accuracy loss ≤ 2 percentage points relative to the same teacher.

| Question | Recorded result | Boundary |
|---|---|---|
| Smaller replacement than the strong J comparison? | First-passing candidate-budget ratio C/J **0.788324**: about **21.2% smaller**; 11 teachers smaller, 4 equal, 1 larger | Paired geometric mean over a finite grid, not a theoretical minimum or whole-model reduction |
| Smaller whole model? | Parameters about **1.26%** lower; NPZ bytes about **1.53–1.56%** lower | Not RAM, VRAM, or deployment-package savings |
| Faster forward pass? | CPU time about **16.33%** lower at batch 1 and **5.67%** at batch 128 | Shared CPU, one thread, fixed synthetic inputs; excludes loading; not a general speed guarantee |
| Successful beyond this setting? | CIFAR-10 original Comp lost **4.26 / 4.22 / 4.38 pp** on the already-evaluated official test | Exceeds the 2 pp quality allowance; the examined improvement series also failed |

[Full results and source links](research-checkpoints/2026-10-27/README.md) · [CIFAR-10 negative results](research-checkpoints/2026-10-27/README.md)

## Recompute, inspect, or reproduce

| Goal | Start here |
|---|---|
| Recalculate published scalar summaries without training | [Saved-result analysis](research/public_saved_analysis_20261027/README.md) |
| Inspect all 25 post-hoc threshold conditions, including censored cases | [Threshold sensitivity](research/threshold_robustness_20261027/README.md) |
| Inspect the source-only pipeline and missing inputs | [Matched L/J/C](research/digits_matched_source_20261027/README.md) → [Budget grid](research/digits_joint_budget_source_20261027/README.md) → [Whole-model storage](research/selected_whole_storage_source_20261027/README.md) → [Latency](research/selected_latency_source_20261027/README.md) |
| Reproduce the separate historical eight-seed baseline | [Historical baseline quickstart](QUICKSTART.md), with its [unresolved discrepancy](docs/REPRODUCTION_DISCREPANCY.md) |

Scalar recomputation, synthetic tests, and artifact hash checks are **not** fresh real-data end-to-end reproduction or independent scientific replication. The source-only packages do not include the original weights, arrays, or dataset. See [current validation status](STATUS.md) before choosing a route.

## Limits and open questions

- **External generalization is unconfirmed.** The separate writer-data acquisition did not complete; there is no model-evaluation result for it.
- The historical 450-example Digits test set was already evaluated. Excluding its predictions from the newer matched/grid work does not make it unseen again.
- Threshold sensitivity is post-hoc: 20 conditions yielded complete ratios, while 5 stricter conditions were censored. Individual selections can sit at the quality boundary.
- CIFAR failure does not identify a unique cause or prove all future composition methods impossible. Loss, capacity, parameterization, and optimization remain unresolved explanations.
- The older baseline reproduction discrepancy, invalidated Phase 2E, and earlier negative cost result remain part of the record. [Corrections and negative results](docs/NEGATIVE_RESULTS.md)

**The present scope is sufficient for a limited research report, not a claim of practical large-scale compression.** Further work needs an identifiable question and independent evidence, not indefinite retuning on already-observed data.

## History, contribution, and citation

This is the `research/checkpoint-20261026` view, not a main release. [Historical research index](docs/RESEARCH_INDEX.md) · [Preserved earlier entry pages](archives/README.md) · [Contributing](CONTRIBUTING.md) · [Citation](CITATION.cff)

Cite the exact commit, protocol, and result artifacts. Historical records and failures are preserved; reorganizing documentation does not change their evidence status.

Original code and documentation: [Apache-2.0](LICENSE). Third-party data and dependencies retain their own licenses.
