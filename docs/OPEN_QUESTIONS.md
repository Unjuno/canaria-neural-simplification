# Open research questions

Experiments are currently paused. This file records the questions that remain genuinely unresolved after the current research program, so future work does not confuse an open question with a failed or already-decided one.

## Priority S — theory-critical

### 1. Codec-independent global complexity

Whole-network FP32/q8/zlib accounting supports a real net reduction after compilation, and measured shell growth offsets only a small fraction of removed-core savings. However, this is not a proof of a codec-independent minimum description length.

Next decisive test: evaluate the same matched models under several independently motivated code families (entropy code, structured factor code, symbolic/operator grammar, residual neural code) and quantify between-code definition uncertainty.

### 2. External validity

Most decisive evidence comes from a digits-like task and an 8-block residual CNN. The current claims must not be promoted to a universal neural-network law before replication on different tasks and graph topologies.

Recommended order: Fashion-MNIST → CIFAR-10 + small ResNet → skip/parallel subgraphs → attention/transformer subgraphs.

### 3. Task-effective repair dimension

Raw trainable parameter count failed to explain adaptive recovery. Equal-capacity local/global adapters did not reproduce the apparent downstream-shell advantage.

Next test: measure the effective rank/spectrum of the trainable-parameter-to-logit Jacobian and compare repair subspaces at matched task-effective dimension rather than matched parameter count.

## Priority A — mechanism

### 4. Why does simplification increase with repair time?

The blinded confirmatory map shows a strong increase in simplification frequency for larger repair budgets. The causal decomposition between intrinsic replaceability, optimizer dynamics, representational reorganization, and spare adaptive capacity is incomplete.

### 5. Recursive compiler fixed points

The strong hypothesis of unlimited recursive simplification was not supported. A task-conditioned nonlinear complexity floor or grammar-dependent fixed point remains plausible.

Next test: repeat recursive compilation with a strictly expanded, preregistered grammar and report whether the terminal description length moves materially.

### 6. Off-manifold versus task-manifold complexity

Current replacement quality is measured mainly on the task distribution and augmentations. It remains open whether a compiled span is simple only on the data manifold.

Required probes: interpolation, strong augmentation, Gaussian/noise controls, adversarial/off-manifold inputs, and synthetic manifolds with known latent coordinates.

### 7. Null models

We still need stronger controls separating learned compositional simplification from generic overparameterized-network behavior.

Priority nulls: random weights, shuffled labels, memorized random labels, synthetic teachers with known composition complexity, and deliberately non-compressible functions.

## Priority B — representation and mechanism dictionaries

### 8. Mechanism algebra

Exploratory results motivate testing closure, idempotence, absorbing elements, and approximate associativity of mechanism composition. This is not yet established.

### 9. Grammar saturation / finite-dictionary hypothesis

Apparent archetype saturation may be a finite candidate-grammar artifact. Expand the grammar in nested stages and test whether the number/dimension of stable modes saturates.

### 10. Cross-seed canonical dictionary

Simple gauge alignment was insufficient. Future work should compare canonicalization, functional alignment, CKA/Jacobian similarity, and tightly capacity-controlled nonlinear adapters against random-dictionary controls.

### 11. Archetype uncertainty

Discrete cluster counts depend on thresholds. Replace single cluster counts with bootstrap stability, intrinsic dimension, density modes, nearest-neighbor graphs, and persistent-topology diagnostics.

## Priority C — learning-cycle applications

### 12. Residual formation and recompilation

New tasks produced residual structure, but the full learn → residual formation → compile → freeze cycle is not decisively established across tasks.

### 13. Distillation

It remains open whether teacher mechanism complexity predicts minimum student capacity better than teacher parameter count. This should be tested with normal KD, compiled-teacher KD, and mechanism-aware distillation under matched optimization budgets.

### 14. Whole-network storage frontier

The exact 9,926-byte whole-network model is independently confirmed, but not globally minimal. Further storage reductions should be treated as engineering/representation experiments unless they also distinguish a theoretical hypothesis.

## Questions that are *not* open anymore in the tested setting

- Canary is **not** a necessary local condition for simplification.
- Pure downstream location is **not** sufficient to explain adaptive repair advantage after capacity/function-class controls.
- Simple parameter count alone is **not** an adequate measure of repair capacity.
- Strong "all local compression is just complexity relocation" is **not** supported by the measured whole-network codecs.
- Unlimited recursive compilation is **not** supported by the current grammar/pilot evidence.

See `CLAIMS_AND_EVIDENCE.md`, `NEGATIVE_RESULTS.md`, and `ROADMAP.md` before proposing new experiments.