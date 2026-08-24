# Generalization status ledger

This file records outcomes under `GENERALIZATION_ROADMAP.md` without rewriting the roadmap's prespecified transfer taxonomy.

| Phase | Architecture/task shift | Status | Main result | Evidence |
|---|---|---|---|---|
| G0 | Public residual-CNN portability control | partial / software curation | core codec and selected historical runners are public; full clean-room reproduction remains incomplete | repository CI + historical summaries |
| G1 | Fashion-MNIST residual CNN | not run | — | — |
| G2 | simple-task architecture panel | not run | — | — |
| **G3** | **digits task: residual CNN -> small ViT** | **A — adapted transfer** | 4 Transformer blocks -> 2 smaller blocks; 60.18% parameter reduction; tau8 utility 0.9685, 95% CI [0.9610, 0.9767]; q8+zlib state-stream reduction 60.53% with q8 utility 0.9670 [0.9580, 0.9766] | `docs/phases/v20/63_G3_SMALL_VIT_GENERALIZATION_PROTOCOL_V20.md`, `64_G3_SMALL_VIT_GENERALIZATION_RESULTS_V20.md` |
| G3b | CIFAR-10 small ViT | not run | — | — |
| G4 | CIFAR-10 small ResNet | not run | — | — |
| **G5** | **image-token ViT -> discrete non-image Transformer encoder** | **Z — zero-shot transfer** | frozen G3 4-block -> 2 smaller-block compiler, no new candidate search; tau0 utility 0.99184, 95% CI [0.97986, 1.00199], 8/8 PASS95; q8+zlib state-stream reduction 58.47% [58.26%, 58.69%], q8 utility 0.99683 [0.98402, 1.00741] | `docs/phases/v21/66_G5_SEQUENCE_TRANSFORMER_CONFIRM_PROTOCOL_V21.md`, `67_G5_SEQUENCE_TRANSFORMER_RESULTS_V21.md` |
| G6 | small decoder-only LM | not run | — | — |
| G7 | recurrent/state-space control | not run | — | — |
| G8 | arbitrary subgraphs | not run | — | — |

## Interpretation through G5

G3 rejected **zero-shot compiler universality** when first moving from the residual-CNN setting into a ViT: bounded architecture-aware adaptation was required. However, after that Transformer-family condition was fixed, G5 showed **zero-shot transfer within the Transformer family** from image-token classification to a controlled non-image sequence encoder.

This weakens the explanation that the observed simplification is specific to convolutional residual blocks or image patch tokens. It is positive evidence for U1 (phenomenon universality) and limited evidence toward U3 (a reusable Transformer-family adaptation rule), while U2 (one unchanged compiler across all architecture families) remains unsupported.

G5 is still a synthetic sequence-order classification task. It is **not** evidence for natural-language or autoregressive Transformers. Causal masking, next-token loss, generation drift, KV-cache behavior, and long-context dependence remain untested.

## Next highest-information test

**G6 — small decoder-only causal language model.** Measure teacher-forced next-token loss/perplexity, sequence-length dependence, multi-step generation drift, and whole-model state/code reduction under a frozen compiler condition.
