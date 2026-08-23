# Research roadmap (experiments resumed for generalization testing)

The authoritative list of unresolved questions, including questions that are **no longer open**, is [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md).

## Primary research axis: cross-architecture generalization

Before optimizing the current residual-CNN system further, the highest-value question is whether task-conditioned simplification transfers to substantially different network families, and whether transfer is zero-shot, requires bounded family-specific adaptation, is conditional, or fails under a fair test budget.

See:

- [`GENERALIZATION_ROADMAP.md`](GENERALIZATION_ROADMAP.md) — prespecified transfer taxonomy and experimental sequence.
- [`GENERALIZATION_STATUS.md`](GENERALIZATION_STATUS.md) — live outcome ledger.

The roadmap distinguishes:

1. **phenomenon universality** — simplification recurs across network families;
2. **compiler universality** — one unchanged compiler works everywhere;
3. **adaptation-rule universality** — a small set of explicit family-specific rules is sufficient.

It is acceptable, and scientifically useful, for some networks to simplify only after adaptation or not to simplify under the tested budget.

## Completed generalization milestone

### G3 — small ViT architecture shift: **A — adapted transfer**

Holding the sklearn-digits task fixed, a 4-block small ViT was compiled into 2 smaller Transformer blocks.

Confirmatory n=8 result:

- whole-model parameter reduction: **60.18%**;
- zero-shot tau0 utility: **0.9421**, 95% CI **[0.9321, 0.9514]** -> FAIL;
- tau2 utility: **0.9613**, 95% CI **[0.9548, 0.9676]**;
- tau8 utility: **0.9685**, 95% CI **[0.9610, 0.9767]** -> adapted-transfer PASS;
- q8+zlib whole-model reduction: **60.53%**, 95% CI **[60.38%, 60.68%]**;
- q8 matched-control utility: **0.9670**, 95% CI **[0.9580, 0.9766]**.

This is evidence against the claim that the phenomenon is restricted to the original residual-CNN architecture family, while also showing that the compiler does **not** transfer zero-shot under the tested recipe.

Evidence:
- [`docs/phases/v20/63_G3_SMALL_VIT_GENERALIZATION_PROTOCOL_V20.md`](phases/v20/63_G3_SMALL_VIT_GENERALIZATION_PROTOCOL_V20.md)
- [`docs/phases/v20/64_G3_SMALL_VIT_GENERALIZATION_RESULTS_V20.md`](phases/v20/64_G3_SMALL_VIT_GENERALIZATION_RESULTS_V20.md)

## Highest-priority next experiments

1. **G1 — Fashion-MNIST residual CNN**: dataset shift with architecture kept near the original reference. Use a separate eligibility/validation split and untouched final test set.
2. **G3b — CIFAR-10 small ViT**: test whether the Transformer result survives a substantially harder image distribution.
3. **G5 — sequence Transformer encoder**: remove image-patch-specific structure while retaining bidirectional Transformer computation.
4. **G6 — small decoder-only language model**: test causal attention, autoregressive error accumulation, and context-length dependence.
5. **ViT mechanism decomposition**: attention-only, MLP-only, full-block, and multi-block replacement to test whether composition subadditivity itself transfers.

## Other high-information questions

1. **Codec-independent complexity** — compare several independently motivated MDL/code families and quantify definition uncertainty.
2. **Task-effective repair dimension** — replace raw parameter count with the effective rank/spectrum of the trainable-parameter-to-logit Jacobian.
3. **Off-manifold complexity** — separate task-manifold simplification from full-input-space approximation.
4. **Null models** — random weights/labels, memorization controls, known-complexity synthetic teachers, and deliberately non-compressible functions.
5. **Mechanism algebra** — closure, associativity, idempotence, absorbing mechanisms, and stability under grammar expansion.
6. **Cross-seed canonicalization** — representation/functional alignment before claiming a global finite mechanism dictionary.
7. **Residual formation / recompilation** — test the learn -> residual formation -> compile -> freeze cycle on new tasks.
8. **Distillation** — test whether teacher mechanism complexity predicts minimum student capacity better than teacher parameter count.
9. **Whole-network storage frontier** — now that 9,926 B is confirmed, test whether 8–9 KB can be reached without sacrificing independent-seed stability; this remains lower priority than external validity.

Do not prioritize additional seeds solely to make an already decisive confirmatory result more significant. Prefer new conditions that distinguish competing explanations, and do not retune a supposedly confirmatory architecture cohort after inspecting its outcomes.
