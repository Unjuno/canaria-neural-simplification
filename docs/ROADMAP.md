# Research roadmap (experiments currently paused)

The repository is currently in a curation/reuse phase. The authoritative list of unresolved questions, including questions that are **no longer open**, is [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md).

When experiments resume, the highest-information tests are:

1. **External validity** — Fashion-MNIST, CIFAR-10 + small ResNet, then arbitrary skip-graph/subgraph cuts.
2. **Codec-independent complexity** — compare several independently motivated MDL/code families and quantify definition uncertainty.
3. **Task-effective repair dimension** — replace raw parameter count with the effective rank/spectrum of the trainable-parameter-to-logit Jacobian.
4. **Off-manifold complexity** — separate task-manifold simplification from full-input-space approximation.
5. **Null models** — random weights/labels, memorization controls, known-complexity synthetic teachers, and deliberately non-compressible functions.
6. **Mechanism algebra** — closure, associativity, idempotence, absorbing mechanisms, and stability under grammar expansion.
7. **Cross-seed canonicalization** — representation/functional alignment before claiming a global finite mechanism dictionary.
8. **Residual formation / recompilation** — test the learn → residual formation → compile → freeze cycle on new tasks.
9. **Distillation** — test whether teacher mechanism complexity predicts minimum student capacity better than teacher parameter count.
10. **Whole-network storage frontier** — now that 9,926 B is confirmed, test whether 8–9 KB can be reached without sacrificing independent-seed stability.

Do not prioritize additional seeds solely to make an already decisive confirmatory result more significant; prefer new conditions that distinguish competing explanations.