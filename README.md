# Canaria

**Research on simplifying neural computation while retaining task-relevant behavior.**

[日本語](README.ja.md) · [Research families](docs/RESEARCH_INDEX.md) · [Digits / CIFAR assessment](RESEARCH_CHECKPOINT.md) · [Reproduction status](STATUS.md) · [Repository layout](REPOSITORY_LAYOUT.md)

What changes when a network span is approximated as a composed input–output function rather than block by block? Canaria investigates learned replacement, consolidation during training, precision and structured representations, recursive composition, vision/sequence models, and runtime prototypes. The repository preserves **conditions where simplification works and conditions where quality is not retained**.

## Three research directions

1. **Post-training replacement:** compare local, jointly trained, and composed mappings; distinguish replacement size from whole-model parameters, serialized bytes, and execution time.
2. **Training-time consolidation:** study learning and repair after changing computation. Immediate replacement and performance after adaptation are different results. [Training-time studies](docs/TRAINING_TIME_CONSOLIDATION.md)
3. **Task preservation and its limits:** lower feature error does not necessarily preserve classification or generation. Objective, correction-subspace, recursive, and cross-family studies examine these boundaries.

## Additional studies across the project

These are recorded primary reports/tables, not new measurements or reproductions performed during this organization. These families have different models, comparisons, and quality gates; their cohorts are not pooled.

| Separate recorded finding | Reduction or improvement | Conditions and evidence |
|---|---|---|
| Historical whole-model / codec | v18 FP32 code **−26.138%**, q8 zlib code **−28.786%**; v19 fixed payload **9,926 bytes**, zero round-trip logit difference | Matched repair control, fixed architecture/codec; not total executable or RAM. [v18](archives/research-history/results/v18/raw/phaseX_summary.json) / [v19](archives/research-history/results/v19/raw_AD/confirmatory_codec_summary.json) |
| SmallViT and synthetic sequence models | Parameter reductions **60.182%** in v20 SmallViT, **58.053%** in v21 sequence, **58.154%** in v22 causal LM | v20/v22 reported quality passes require repair; v21 passes after activation fitting without task repair. Immediate v20 utility and v22 generation gates fail. [v20](archives/research-history/results/v20/g3_confirmatory_summary.json) / [v21](archives/research-history/results/v21/g5_confirmatory_summary.json) / [v22](archives/research-history/results/v22/g6_confirmatory_summary.json) |
| California Housing regression | Composed/component-wise selected replacement budget ratio **0.476029**; test R² difference **+0.006858** | Prospective 8-model-seed study, one fixed split and residual-MLP grammar. [Report](https://github.com/Unjuno/canaria-neural-simplification/blob/f7d23c9dad2788e64b94f2a268fe3e0b1084d77c/results/phase4/california_regression/stage_b_confirmatory/RESULT.md) |
| Staged training-time consolidation | G7 improves final PPL over early/late consolidation; G19 staged PPL 19.3580 versus direct 20.0475 at 192 updates each | Small character-LM records. Without intervening learning, G17 establishes equivalence to direct fitting within the prespecified ± 0.10 PPL band. [Training](docs/TRAINING_TIME_CONSOLIDATION.md), [later controls](docs/LATE_STAGE_FINDINGS.md) |
| Direction-aware task-damage prediction | G23–G25 record direction/logit-error prediction and transfer; G26 improves future-horizon correction | Public summaries and provenance hashes, not independently reverified original observations/predictor implementation. Prediction is not a solved optimal controller. [Results](docs/LATE_STAGE_FINDINGS.md), [retained artifacts](results/training_time/ARTIFACT_INVENTORY.md) |

In contrast, examined small real-text LM protocols failed rollout preservation despite near-preserved teacher-forced perplexity. Precision/repair work also includes invalidated implementation results; the [correction history](results/phase2/precision_composition/INVALIDATED_HISTORY.md) remains part of the evidence.

## Principal results: name the comparator

The current Digits comparison replaces the first two blocks of a four-block residual MLP while retaining the rest. **L** fits two maps locally; **J** jointly trains two maps; **C** uses one map. Sixteen teachers share one dataset split—not sixteen datasets. Selection requires validation endpoint NMSE ≤ 0.08 and accuracy loss ≤ 2 percentage points relative to the teacher.

| Comparison and object | Recorded result | Scope |
|---|---|---|
| **Original teacher → C / J: whole model** | C has **about 44.14% fewer parameters**, J **about 43.43% fewer**. Teacher 71,626; C 39,754–41,290; J 39,754–42,314 | Geometric mean of 16 per-teacher ratios. Selected quality allowance is not identical teacher accuracy |
| **L → J: matched-budget training regime** | NMSE lower **16/16**; mean accuracy **+1.76 pp**, 95% CI [+1.16, +2.34] | Same initial parameters/600 updates; teacher forcing and gradient paths also differ, not an objective-only causal effect |
| **J → C: replacement** | First-passing tested budget **about 21.2% lower**; C/J 0.788324, 95% CI [0.694450, 0.904368] | Finite grid; 11 smaller, 4 tied, 1 larger. Not a theoretical minimum |
| **J → C: whole model** | Additional reduction **about 1.26% parameters**, **1.53–1.56% NPZ bytes** | Different from teacher-relative reduction; serialized bytes are not RAM/VRAM |
| **J → C: CPU forward** | **About 16.33% shorter** at batch 1, **5.67%** at batch 128 | Shared CPU, one thread, fixed synthetic inputs; excludes loading and GPU claims |
| **CIFAR-10: quality gate** | Original Comp official-test accuracy losses **4.26/4.22/4.38 pp** | Exceeds 2 pp allowance; examined improvement/diagnostic series also missed the target. Not impossibility of the method |

[Methods, confidence intervals and evidence](research-checkpoints/2026-10-27/README.md) · [Whole-model storage](research/public_checkpoint_20261027/research/selected_whole_storage/REPORT_JA.md) · [CPU conditions](research/public_checkpoint_20261027/research/selected_latency/REPORT_JA.md)

**Compression relative to the teacher and C's incremental advantage over the strong J comparator are different questions.** Neither “the whole model barely shrinks” without a comparator nor “21% whole-model reduction” describes this evidence correctly.

### Family map

- **Baseline and consolidation:** [historical Digits](docs/CORE_DISCOVERY_REPLICATION_DIGITS.md), [training-time](docs/TRAINING_TIME_CONSOLIDATION.md), [later controls](docs/LATE_STAGE_FINDINGS.md).
- **Precision, structure and encoding:** [corrected precision studies](docs/phase2/README.md), [v10–v25 archives](archives/README.md); distinguish core code, whole parameters and actual bytes.
- **Vision, sequence and language models:** [SmallViT comparison](docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md), [branch-specific extensions](docs/RESEARCH_INDEX.md); distinguish immediate from adapted performance.
- **Recursive composition, interfaces and shift:** [C-series and QR](docs/RESEARCH_INDEX.md); correction-subspace dimension is not whole-model compression.
- **Regression and external validity:** [family evidence](docs/RESEARCH_INDEX.md); teacher eligibility and independent datasets matter.
- **Runtime systems:** [CPU PoC](docs/RUNTIME_POC.md), [S1–S7](docs/RESEARCH_INDEX.md); application-managed arenas are not total-process memory.

### Reproduction is a separate result

The [R87 tested recipe and fresh confirmation](https://github.com/Unjuno/canaria-neural-simplification/blob/56a8d1a50f15cad0d930cc899336071fa8efe7dc/results/reproduction/r87_resolution/RESULT.md) reports numerical agreement on tested Intel/AMD hosts and a separate 16-model-seed composed/component-wise budget ratio 0.531610. It is neither the current C/J comparison nor recovery of the historical 8-seed archive. The [historical exact-reproduction discrepancy](docs/REPRODUCTION_DISCREPANCY.md) remains separate.

This public snapshot preserves the Phase2E `INVALIDATED_IMPLEMENTATION_BUG` record and its correction history; an invalidated run must not be used as scientific evidence.

## What is not established

The evidence supports conditional reductions and improvements, not a universal compression law or general deployment savings. California Housing and cross-model studies provide evidence beyond the original setting, but do not establish generality across unseen datasets or large architectures. Better hidden approximation need not improve task behavior. Read [negative results and corrections](docs/NEGATIVE_RESULTS.md) alongside positive findings.

## Inspect, recompute, reproduce

- [Saved-scalar recomputation](research/public_saved_analysis_20261027/README.md), [threshold sensitivity including censored cases](research/threshold_robustness_20261027/README.md).
- Source-only packages: [matched](research/digits_matched_source_20261027/README.md), [grid](research/digits_joint_budget_source_20261027/README.md), [storage](research/selected_whole_storage_source_20261027/README.md), [latency](research/selected_latency_source_20261027/README.md). Check each package's missing inputs.
- [Historical baseline Quickstart](QUICKSTART.md), [reproduction boundaries](docs/REPRODUCIBILITY.md), [documentation](docs/README.md).

This page organizes recorded work; it is not a new experimental run. Static inspection, saved-summary recomputation, technical reruns and independent scientific replication are distinct. Original protocols, failures, corrections and branch evidence are preserved.
