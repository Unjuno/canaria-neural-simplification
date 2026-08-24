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
| **G5** | **image-token ViT -> discrete non-image Transformer encoder** | **Z — zero-shot transfer** | frozen G3 4-block -> 2 smaller-block compiler; tau0 utility 0.99184 [0.97986, 1.00199]; q8+zlib state-stream reduction 58.47% [58.26%, 58.69%] | `docs/phases/v21/66_G5_SEQUENCE_TRANSFORMER_CONFIRM_PROTOCOL_V21.md`, `67_G5_SEQUENCE_TRANSFORMER_RESULTS_V21.md` |
| **G6** | **Transformer encoder -> causal decoder-only next-token model** | **A — adapted transfer** | 4 causal blocks -> 2 smaller causal blocks; 58.15% parameter reduction. tau0 PPL utility 0.9588 [0.9507, 0.9669] but generation utility 0.8359 [0.8029, 0.8699], so zero-shot FAIL. tau8 PPL utility 0.9852 [0.9799, 0.9895] and generation utility 0.9632 [0.9467, 0.9770], so adapted PASS. q8+zlib state-stream reduction 57.88% [57.78%, 57.97%]. | `docs/phases/v22/69_G6_DECODER_LM_CONFIRM_PROTOCOL_V22.md`, `70_G6_DECODER_LM_RESULTS_V22.md` |
| G7 | recurrent/state-space control | not run | — | — |
| G8 | arbitrary subgraphs | not run | — | — |

## Interpretation through G6

G3 rejected zero-shot compiler universality when first moving from residual CNNs to ViT, but established adapted transfer. G5 then showed that the fixed Transformer-family compiler transfers zero-shot from image tokens to a controlled non-image sequence encoder.

G6 adds a stricter causal/autoregressive test. The same 4->2 causal-block simplification preserved teacher-forced perplexity well enough to meet the zero-shot PPL threshold, but **free-running generation accumulated errors and failed the preregistered zero-shot generation threshold**. Bounded tau=8 shell repair restored both PPL and generation utility, producing an adapted-transfer PASS.

This is important evidence that decoder models require evaluation beyond teacher-forced loss. A simplification that appears acceptable under PPL/token accuracy can still fail during autoregressive rollout.

The combined G3/G5/G6 record weakens explanations based on convolution, image patches, or bidirectional encoder attention alone. It supports U1 (phenomenon transfer) and partial U3 (reusable family-specific adaptation), while one universal zero-shot compiler remains unsupported.

G6 remains a small **synthetic prompted language**, not natural language or a pretrained LLM. Long context, KV-cache behavior, sampling, arbitrary prompts, and pretrained-model transfer remain open.

## Next highest-information tests

1. Small real-text / natural-language decoder LM.
2. Generation-length sweep for error amplification.
3. Attention-only vs MLP-only causal compilation.
4. Pretrained small language model with independent fine-tuning/checkpoint replicates.
