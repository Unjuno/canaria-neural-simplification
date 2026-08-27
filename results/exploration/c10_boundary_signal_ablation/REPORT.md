# C10 boundary re-alignment signal ablation — exploratory report

C10 asks what supervision signal is required when a recursively generated `C12 -> C34` hierarchy is re-opened before the next recursive compile.

Fresh exploratory seeds: `1410–1412`. Held-out test was not evaluated.

## Aggregate validation results

| top-boundary signal | hierarchy NMSE vs original | final NMSE vs original | final/direct ratio | final val acc |
|---|---:|---:|---:|---:|
| frozen | 0.06951 | 0.07217 | 1.742× | 0.9580 |
| **hidden MSE** | **0.03412** | **0.04862** | **1.173×** | 0.9654 |
| logit distillation | 0.06870 | 0.08046 | 1.947× | 0.9654 |
| label CE | 1.39480 | 1.36978 | 33.260× | 0.9605 |
| direct original single | — | 0.04146 | 1.000× | 0.9679 |

Teacher mean validation accuracy: `0.9753`.

The final-NMSE ordering was identical in all three seeds:

`hidden_mse < frozen < logit_distill < label_ce`.

## Interpretation

The experiment separates two notions that had previously been conflated:

1. **downstream task utility**, and
2. **hidden-space fidelity suitable for further recursive compilation**.

`logit_distill` produced final validation accuracy essentially equal to `hidden_mse`, but its final hidden-space NMSE was worse than leaving the hierarchy frozen. Thus matching the downstream logits was not enough to restore the internal representation required by the tested recursive compiler grammar.

`label_ce` is even more striking: validation accuracy remained near the other replacement conditions, while final hidden-space NMSE became extremely large. The label objective can move the hierarchy into a task-useful but representation-incompatible solution.

In this testbed, the original hidden-state target is therefore doing more than preserving task behavior: it acts as a **canonical interface constraint** between recursive compilation levels.

## Boundary

C10 is exploratory and validation-only. It does not establish that hidden targets are universally necessary. It motivates the next question: how much hidden-state information is actually required? A compressed sketch or partial interface target may be sufficient.
