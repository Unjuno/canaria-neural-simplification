# Generalization status ledger

This file records outcomes under `GENERALIZATION_ROADMAP.md` without rewriting the roadmap's prespecified transfer taxonomy.

| Phase | Architecture/task shift | Status | Main result | Evidence |
|---|---|---|---|---|
| G0 | Public residual-CNN portability control | partial / software curation | core codec and selected historical runners are public; full clean-room reproduction remains incomplete | repository CI + historical summaries |
| G1 | Fashion-MNIST residual CNN | not run | — | — |
| G2 | simple-task architecture panel | not run | — | — |
| **G3** | **digits task: residual CNN -> small ViT** | **A — adapted transfer** | 4 Transformer blocks -> 2 smaller blocks; 60.18% parameter reduction; tau8 utility 0.9685 [0.9610, 0.9767]; q8+zlib state-stream reduction 60.53% | `docs/phases/v20/63_G3_SMALL_VIT_GENERALIZATION_PROTOCOL_V20.md`, `64_G3_SMALL_VIT_GENERALIZATION_RESULTS_V20.md` |
| G3b | CIFAR-10 small ViT | not run | — | — |
| G4 | CIFAR-10 small ResNet | not run | — | — |
| **G5** | **image-token ViT -> discrete non-image Transformer encoder** | **Z — zero-shot transfer** | frozen G3 4-block -> 2 smaller-block compiler; tau0 utility 0.99184 [0.97986, 1.00199]; q8+zlib state-stream reduction 58.47% | `docs/phases/v21/66_G5_SEQUENCE_TRANSFORMER_CONFIRM_PROTOCOL_V21.md`, `67_G5_SEQUENCE_TRANSFORMER_RESULTS_V21.md` |
| **G6** | **Transformer encoder -> synthetic causal decoder-only next-token model** | **A — adapted transfer** | tau0 PPL utility passed but free-running generation failed; tau8 repair restored both. 58.15% parameter reduction; q8+zlib state-stream reduction 57.88%. | `docs/phases/v22/69_G6_DECODER_LM_CONFIRM_PROTOCOL_V22.md`, `70_G6_DECODER_LM_RESULTS_V22.md` |
| **G6b** | **synthetic causal language -> held-out natural English character LM** | **N — no transfer under tested budget** | 52.28% parameter reduction. tau0 PPL utility **0.99703 [0.99576, 0.99816]** but greedy rollout agreement only **0.63265 [0.55029, 0.72689]**. Prespecified tau8 joint repair also failed. | `docs/phases/v23/76_G6B_REALTEXT_CONFIRM_PROTOCOL_V23.md`, `77_G6B_REALTEXT_LM_RESULTS_V23.md` |
| **G6c** | **natural-English LM: hidden-state compiler -> teacher-forced logit-aware compiler** | **N — no transfer under tested budget** | O1 hidden-MSE + teacher-logit KL was selected on discovery seeds, but fresh seeds retained PPL while failing trajectory fidelity: tau0 PPL utility **0.99668 [0.99580, 0.99761]**, rollout agreement **0.56169 [0.48681, 0.63737]**. tau8 joint repair also failed. | `docs/phases/v24/80_G6C_LOGIT_AWARE_PILOT_PROTOCOL_V24.md`, `81_G6C_LOGIT_AWARE_CONFIRM_PROTOCOL_V24.md`, `82_G6C_LOGIT_AWARE_RESULTS_V24.md` |
| **G6d** | **natural-English LM: teacher-prefix KL -> one-iteration on-policy trajectory distillation** | **N — no transfer under tested budget** | T2 used compiler-generated prefixes out to 24 steps at fixed 52.28% reduction. Fresh-seed PPL utility stayed **0.99709 [0.99627, 0.99784]**, but rollout agreement was only **0.61963 [0.51123, 0.72900]**, with large seed heterogeneity (0.342–0.882). | `docs/phases/v25/83_G6D_TRAJECTORY_AWARE_PILOT_PROTOCOL_V25.md`, `84_G6D_TRAJECTORY_AWARE_CONFIRM_PROTOCOL_V25.md`, `85_G6D_TRAJECTORY_AWARE_RESULTS_V25.md`, `results/v25/g6d_confirmatory_summary.json` |
| G7 | recurrent/state-space control | not run | — | — |
| G8 | arbitrary subgraphs | not run | — | — |

## Interpretation through G6d

The transfer map remains explicitly **mixed**.

- G3: adapted transfer into ViT.
- G5: zero-shot transfer within the Transformer family to a controlled non-image encoder.
- G6: synthetic causal decoding requires bounded repair.
- G6b: natural English exposes a stronger autoregressive trajectory boundary.
- G6c: teacher-forced logit awareness alone does not remove that boundary.
- **G6d: one bounded dataset-aggregation pass on compiler-generated prefixes is also insufficient to make trajectory fidelity stable across fresh seeds.**

Across v23–v25, the repeated pattern is unusually clear: a 52%-smaller compiler can preserve teacher-forced likelihood nearly perfectly while free-running trajectories diverge. The failure cannot now be attributed solely to hidden-state fitting, omission of output logits, or complete absence of student-prefix exposure.

The v25 seed range (rollout agreement 0.342–0.882) suggests a second question alongside stronger adaptation: **what dynamical property predicts whether a trained teacher/compiler pair is trajectory-stable under simplification?** Post-hoc n=8 correlations are exploratory only and are not treated as evidence.

## Strong evaluation lesson

For autoregressive models, **teacher-forced likelihood is necessary but not sufficient**. PPL utility around 0.997 can coexist with major rollout divergence after aggressive internal simplification.

## Next highest-information tests

1. **Capacity-vs-stability frontier:** reduce compression aggressiveness while holding the trajectory-aware objective fixed, to distinguish an objective failure from a genuine minimum-capacity boundary.
2. **Iterative dataset aggregation:** regenerate on-policy prefixes after each bounded refinement stage, but only under a separately frozen adaptation budget.
3. Measure teacher entropy/margins and local error amplification as candidate predictors of stable vs unstable seeds.
4. Small pretrained/subword LM only after a natural-text causal condition becomes stable on this controlled model.
5. G4/G3b CIFAR-10 ResNet/ViT to continue non-language external validity.
