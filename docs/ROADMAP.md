# Research roadmap (experiments currently paused)

The repository is currently in a curation/reuse phase. The authoritative list of unresolved questions, including questions that are **no longer open**, is [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md).

## Primary next research axis: cross-architecture generalization

Before optimizing the current residual-CNN system further, the highest-value question is whether task-conditioned simplification transfers to substantially different network families, and whether transfer is zero-shot, requires bounded family-specific adaptation, is conditional, or fails under a fair test budget.

See the dedicated roadmap:

- [`GENERALIZATION_ROADMAP.md`](GENERALIZATION_ROADMAP.md) — CNN → ViT/Transformer → small language model → recurrent/state-space → arbitrary subgraphs, with explicit adaptation budgets and Z/A/C/N/I transfer labels.

The roadmap deliberately distinguishes:

1. **phenomenon universality** — simplification recurs across network families;
2. **compiler universality** — one unchanged compiler works everywhere;
3. **adaptation-rule universality** — a small set of explicit family-specific rules is sufficient.

It is acceptable, and scientifically useful, for some networks to simplify only after adaptation or not to simplify under the tested budget.

## Other high-information questions

1. **External validity / transfer map** — execute the cross-architecture roadmap, prioritizing Fashion-MNIST residual CNN, small ViT, CIFAR-10 ResNet/ViT, then sequence Transformers.
2. **Codec-independent complexity** — compare several independently motivated MDL/code families and quantify definition uncertainty.
3. **Task-effective repair dimension** — replace raw parameter count with the effective rank/spectrum of the trainable-parameter-to-logit Jacobian.
4. **Off-manifold complexity** — separate task-manifold simplification from full-input-space approximation.
5. **Null models** — random weights/labels, memorization controls, known-complexity synthetic teachers, and deliberately non-compressible functions.
6. **Mechanism algebra** — closure, associativity, idempotence, absorbing mechanisms, and stability under grammar expansion.
7. **Cross-seed canonicalization** — representation/functional alignment before claiming a global finite mechanism dictionary.
8. **Residual formation / recompilation** — test the learn → residual formation → compile → freeze cycle on new tasks.
9. **Distillation** — test whether teacher mechanism complexity predicts minimum student capacity better than teacher parameter count.
10. **Whole-network storage frontier** — now that 9,926 B is confirmed, test whether 8–9 KB can be reached without sacrificing independent-seed stability; this is lower priority than architecture generalization.

Do not prioritize additional seeds solely to make an already decisive confirmatory result more significant. Prefer new conditions that distinguish competing explanations, and do not retune a supposedly confirmatory architecture cohort after inspecting its outcomes.
