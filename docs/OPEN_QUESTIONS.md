# Open research questions

**Project mode: public-snapshot / handoff.**

The purpose of this file is to leave bounded questions that another researcher can pick up. It is not a commitment to continue expanding the current project.

## Priority 0 — closure tasks before a final snapshot

These are the only experiments currently considered potentially necessary for closing public claims.

### 1. Clean-repository reproduction

Clone the public repository into a clean environment and reproduce at least one representative confirmatory pipeline without relying on unpublished `/mnt/data` paths.

Pass criterion: the public instructions and retained artifacts are sufficient to reproduce the stated decision outcome.

### 2. Direct compositional-simplification replication on a clearly different family

Only needed if a publication-level novelty claim requires stronger external validity.

The decisive design should compare:

- component-wise simplification;
- composed-span simplification;
- matched replacement/optimization budget;
- matched task-utility criterion;
- fresh seeds;
- a clearly different architecture/task family.

The point is to test the core discovery directly, not to optimize another controller.

### 3. Minimal runtime-compilation proof-of-concept

Only needed if systems/deployment claims are made.

Measure:

- compact serialized bytes;
- load/compile/materialization time;
- peak RAM/VRAM;
- inference latency/throughput;
- task utility/fidelity.

A negative result is acceptable. The question is whether a compact functional representation can actually be used as a deployment artifact and which resource dimension, if any, improves.

## Scientific questions left for future researchers

### 4. Grammar-independent complexity

Current composition and codec results are operational. How stable are the conclusions across independently motivated description grammars?

### 5. Large pretrained models

Do compositional simplification and training-time recontracting occur in pretrained subword language models at useful scales?

### 6. Functional boundaries

Can useful replacement boundaries be detected automatically without enumerating many spans?

### 7. Why does recontracting improve compiler optimization?

Current evidence shows lower normalized fit cost after task learning, but candidate causes remain:

- representation redistribution;
- improved conditioning;
- basin migration;
- architecture curriculum;
- task-manifold dimensionality changes.

### 8. Why does task sensitivity increase at the same time?

G20e/G22 show that normalized fit gets easier while immediate task damage at matched error increases.

Open decomposition:

- downstream Jacobian spectrum;
- LayerNorm effects;
- logit-margin changes;
- error-direction alignment;
- higher-order curvature.

### 9. Risk-model universality

The first/second-order immediate-damage proxy and horizon model transferred across two depth paths in the same small-LM family. Test different widths, heads, tasks, and architectures before treating coefficients or forms as general laws.

### 10. Cost-aware autonomous control

G21 and G27 show that accurate risk prediction does not by itself yield a Pareto-optimal policy.

A future controller must explicitly value both:

- expected final task damage;
- marginal compiler/training/deployment cost.

### 11. Off-manifold versus task-manifold simplification

How much of the observed simplification is specific to the task/data manifold?

Useful probes include strong augmentation, interpolation, adversarial/off-manifold inputs, and synthetic tasks with known latent structure.

### 12. Null models

Stronger nulls remain valuable:

- random weights;
- shuffled labels;
- memorized random labels;
- synthetic teachers with known compositional complexity;
- deliberately non-compressible functions.

### 13. Recursive fixed points

Unlimited recursive collapse is not supported. Is there a stable task-conditioned complexity floor, and how grammar-dependent is it?

### 14. Deployment representation

Could a future `model.canaria` artifact act as a functional IR compiled differently for GPU, CPU, NPU, or low-memory streaming execution?

See `APPLICATIONS.md`.

## Questions that are not open in the tested settings

- Canary is not a necessary local condition for simplification.
- Teacher-forced PPL is not enough to certify autoregressive functional equivalence.
- Merely splitting a direct compiler fit into two stages without task learning does not reproduce the staged benefit.
- Hard shadow-damage vetoes can block successful final contraction.
- The same normalized functional-error threshold is not equally task-safe before and after recontracting.
- A fixed future-risk cap is not enough to produce an automatic cost/utility Pareto improvement.

Before starting new work, read `CORE_DISCOVERY.md`, `CLAIMS_AND_EVIDENCE.md`, `NEGATIVE_RESULTS.md`, `TRAINING_TIME_CONSOLIDATION.md`, and `LATE_STAGE_FINDINGS.md`.
