# Generalization status ledger

This file records outcomes under `GENERALIZATION_ROADMAP.md` without rewriting the roadmap's prespecified transfer taxonomy.

| Phase | Architecture/task shift | Status | Main result | Evidence |
|---|---|---|---|---|
| G0 | Public residual-CNN portability control | partial / software curation | core codec and selected historical runners are public; full clean-room reproduction remains incomplete | repository CI + historical summaries |
| G1 | Fashion-MNIST residual CNN | not run | — | — |
| G2 | simple-task architecture panel | not run | — | — |
| **G3** | **digits task: residual CNN -> small ViT** | **A — adapted transfer** | 4 Transformer blocks -> 2 smaller blocks; 60.18% parameter reduction; tau8 utility 0.9685, 95% CI [0.9610, 0.9767]; q8+zlib whole-model reduction 60.53% with q8 utility 0.9670 [0.9580, 0.9766] | `docs/phases/v20/63_G3_SMALL_VIT_GENERALIZATION_PROTOCOL_V20.md`, `64_G3_SMALL_VIT_GENERALIZATION_RESULTS_V20.md` |
| G3b | CIFAR-10 small ViT | not run | — | — |
| G4 | CIFAR-10 small ResNet | not run | — | — |
| G5 | sequence Transformer encoder | not run | — | — |
| G6 | small decoder-only LM | not run | — | — |
| G7 | recurrent/state-space control | not run | — | — |
| G8 | arbitrary subgraphs | not run | — | — |

## G3 interpretation

G3 rejects **zero-shot compiler universality** under the tested recipe (tau0 mean utility 0.9421; CI lower bound <0.95), but supports **phenomenon transfer after bounded architecture-aware adaptation**. The result therefore supports U1 more than U2: simplification is observed in a Transformer family, but the CNN recipe does not transfer unchanged.

The task/data distribution is still the sklearn digits task, so this is architecture-family transfer rather than dataset external validity.
