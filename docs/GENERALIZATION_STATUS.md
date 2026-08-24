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
| **G6b** | **synthetic causal language -> held-out natural English character LM** | **N — no transfer under tested budget** | 52.28% parameter reduction. tau0 PPL utility **0.99703 [0.99576, 0.99816]** but greedy rollout agreement only **0.63265 [0.55029, 0.72689]**. Prespecified tau8 joint repair also failed: PPL utility **0.94577 [0.94115, 0.95004]**, rollout agreement **0.41130 [0.33887, 0.48551]**. | `docs/phases/v23/76_G6B_REALTEXT_CONFIRM_PROTOCOL_V23.md`, `77_G6B_REALTEXT_LM_RESULTS_V23.md`, `results/v23/g6b_confirmatory_summary.json` |
| G7 | recurrent/state-space control | not run | — | — |
| G8 | arbitrary subgraphs | not run | — | — |

## Interpretation through G6b

The transfer map is now explicitly **mixed**, which is more informative than a uniform-positive result.

- G3 showed that the residual-CNN recipe does not transfer unchanged to ViT, but bounded Transformer-specific adaptation exposes simplification.
- G5 showed that the resulting Transformer-family condition transfers zero-shot from image tokens to a controlled non-image encoder task.
- G6 showed that causal autoregression introduces a new failure mode: teacher-forced PPL can look acceptable while free-running generation fails. Bounded repair succeeded on the synthetic causal language.
- **G6b shows a genuine boundary under the tested budget.** On held-out natural English prose, teacher-forced PPL is almost unchanged at tau0, yet the generated trajectory diverges strongly; the bounded adaptation selected on pilots does not repair it and can worsen matched-control PPL/rollout fidelity.

This rejects a simplistic interpretation that "Transformer blocks are generally compressible if PPL is preserved." Autoregressive trajectory stability is an additional constraint. The current evidence supports a conditional universality picture: task-effective simplification exists across several architecture families, but whether it is usable depends on downstream error amplification and the compiler/repair objective.

## v23 rollout-horizon diagnostic

A post-confirmatory diagnostic on the same eight seeds measured greedy token agreement versus rollout horizon.

Tau=0 compiled vs baseline agreement decays monotonically:

- 1 token: **0.9219 [0.8906, 0.9531]**
- 2 tokens: **0.9004 [0.8613, 0.9375]**
- 4 tokens: **0.8408 [0.7813, 0.8945]**
- 8 tokens: **0.7651 [0.6938, 0.8315]**
- 16 tokens: **0.6753 [0.5977, 0.7598]**
- 24 tokens: **0.6326 [0.5508, 0.7288]**

This is evidence for an **error-amplification / trajectory-divergence mechanism**: the compiler is locally close, but small early differences alter future autoregressive inputs and accumulate. The selected tau8 joint repair is worse at every measured horizon. See `docs/phases/v23/78_G6B_ROLLOUT_HORIZON_DIAGNOSTIC_V23.md`, `79_G6B_ROLLOUT_HORIZON_RESULTS_V23.md`, and `results/v23/g6b_horizon_summary.json`.

## Strong evaluation lesson

For autoregressive models, **teacher-forced likelihood is necessary but not sufficient**. v22 and v23 jointly show that local next-token metrics can hide substantial rollout divergence. Any future decoder-LM simplification claim must retain rollout-sensitive metrics as primary evidence.

## Next highest-information tests

1. **Objective adaptation:** compare residual-stream MSE against a preregistered logit/KL-aware (and optionally short-rollout-aware) compiler objective on pilot checkpoints, then freeze and re-confirm on fresh seeds.
2. **Small pretrained/subword LM** on real text with independent fine-tuning/checkpoint replicates, only after the objective is frozen; PPL and rollout metrics remain jointly primary.
3. G4/G3b CIFAR-10 ResNet/ViT to map task difficulty outside language.
4. Attention-only vs MLP-only causal compilation to identify which decoder submodule dominates trajectory sensitivity.
