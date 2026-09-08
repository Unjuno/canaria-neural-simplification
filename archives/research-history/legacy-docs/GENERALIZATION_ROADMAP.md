# Generalization roadmap: where does neural simplification transfer?

## Goal

The next research phase should not ask only whether the current residual-CNN compression recipe works unchanged on every network. That is too strong and also scientifically ambiguous.

The central question is:

> **Is task-conditioned computational simplification a recurring property of trained neural networks, and if so, what structural conditions determine when it appears and what bounded adaptation is required to expose it?**

A network family is allowed to require a different boundary definition, candidate grammar, repair locus, or codec. The important distinction is between a general **phenomenon** and a universal **implementation**.

## Four transfer outcomes

Each architecture/task pair should be assigned one of four substantive outcomes, plus an inconclusive state.

| Label | Meaning |
|---|---|
| **Z — zero-shot transfer** | The frozen generic simplification recipe works without family-specific tuning. |
| **A — adapted transfer** | Simplification appears after a small, preregistered family-specific adaptation. |
| **C — conditional transfer** | Simplification appears only for particular depths, spans, repair budgets, tasks, or submodules. |
| **N — no transfer under tested budget** | No net simplification is found after the prespecified adaptation budget. This is a valid negative result. |
| **I — inconclusive** | Baseline training, optimization, calibration, or statistical power is inadequate to distinguish the hypotheses. |

The goal is **not** to force every model into Z or A. A useful theory may predict both positive and negative regimes.

## Three universality hypotheses

The roadmap separates three hypotheses that should not be conflated.

### U1. Phenomenon universality

Across substantially different trained network families, some internal computations admit shorter task-conditioned replacements while preserving matched-control utility after a bounded repair budget.

### U2. Compiler universality

One fixed candidate grammar / boundary / repair recipe works across families without adaptation.

This is stronger than U1 and is not required for the project to succeed.

### U3. Adaptation-rule universality

Even if the exact compiler is not universal, a small family of explicit adaptation rules predicts how to expose simplification in different architectures.

A plausible long-term result is **U1 supported, U2 rejected, U3 partially supported**.

## What counts as simplification across architectures?

Do **not** require a particular byte target such as 44.5 B or 9,926 B to reproduce on another model. Those numbers are implementation-specific.

The cross-architecture invariant should instead be defined operationally:

1. choose a trained subnetwork or subgraph;
2. replace it with a function from a smaller candidate description class;
3. allow only a bounded, matched repair budget;
4. retain task utility relative to an uncompiled continued-training control;
5. verify that **whole-network** description length decreases under the declared codec/accounting rule;
6. separately measure shell growth / complexity relocation.

Phase-specific protocols should lock exact thresholds before outcomes are inspected. The current residual-CNN work suggests `utility >= 0.95` as a useful default target, but generalization phases should preregister task-appropriate metrics and uncertainty rules.

## Adaptation ladder

To avoid tuning until every architecture succeeds, adaptation must be budgeted and ordered.

### Level 0 — frozen generic recipe

Use the same conceptual recipe as the current project:

- contiguous learned computation span;
- calibration-only replacement fit;
- candidate chosen without using held-out task accuracy;
- bounded shell repair;
- matched continued-training control;
- whole-network accounting.

Only tensor shapes and mechanically necessary interface code may change.

### Level 1 — boundary adaptation

Permit only architecture-natural interfaces, for example:

- CNN / ResNet: block input/output feature maps;
- ViT / Transformer: residual-stream states between blocks;
- recurrent networks: recurrent state transition interfaces.

No candidate-family search yet.

### Level 2 — candidate-grammar adaptation

Choose from a preregistered small menu appropriate to the architecture, for example:

- affine / linear;
- low-rank linear;
- convolutional / separable convolutional;
- structured sparse;
- token-wise linear / low-rank residual maps;
- restricted attention-like maps where necessary.

The menu and search budget must be fixed before the confirmatory cohort.

### Level 3 — repair adaptation

Permit a bounded family-specific repair locus/capacity. Compare against a matched uncompiled control with the same optimization budget.

### Level 4 — storage adaptation

Only after functional transfer is established, optimize quantization, sparsity, support coding, entropy coding, and metadata. Storage tuning should not be used to rescue a function that already fails functionally.

A model that succeeds only after Level 3 or 4 is **adapted transfer**, not zero-shot transfer.

## Experimental sequence

The sequence is designed to change one major axis at a time before combining shifts.

### G0 — portability control

**Purpose:** ensure that the cleaned public implementation reproduces the current residual-CNN phenomenon before using it as the reference compiler.

- Current digits-like task.
- Residual-8 reference architecture.
- Reproduce a small subset of the locked simplification, repair, global-accounting, and serialization results.

This is a software/reproducibility gate, not new scientific evidence.

### G1 — task shift with architecture held approximately fixed

**Primary target:** Fashion-MNIST with a residual CNN close to the current architecture.

Question: is the effect tied to the unusually simple digits manifold?

Classify as Z/A/C/N/I using the adaptation ladder.

### G2 — architecture shift with task held simple

On a simple image task, test a small architecture panel:

- plain MLP;
- shallow residual CNN;
- deeper residual CNN;
- wider residual CNN;
- optionally a non-residual CNN control.

Question: which aspects of depth, width, residualization, and convolution are associated with simplification?

### G3 — Transformer family entry point: small ViT

This is a **high-priority generalization test** because it changes the computational substrate from convolutional residual blocks to attention + MLP blocks while retaining an image-classification setting.

Recommended order:

1. small ViT on Fashion-MNIST or another controlled image task;
2. small ViT on CIFAR-10.

Natural replacement boundaries are residual-stream states between Transformer blocks.

Measure separately:

- attention-only spans;
- MLP-only spans;
- full Transformer blocks;
- multi-block compositions.

This directly tests whether the previously observed composition subadditivity has an analogue in attention-based networks.

### G4 — stronger CNN external validity: CIFAR-10 + small ResNet

Question: does simplification survive a more difficult visual task and a more standard residual architecture?

Compare directly with G3 so task difficulty and architecture family are not confounded into one result.

### G5 — Transformer encoder on sequence data

Use a controlled sequence-classification setting first, followed by a real text-classification setting if the controlled experiment succeeds.

Natural interfaces are residual-stream states. Preserve masking semantics and sequence length explicitly.

Question: is ViT transfer actually a Transformer phenomenon, or still specific to image tokenization/classification?

### G6 — small decoder-only language model

Do not start with a large LLM. Use a small decoder-only Transformer where seed-level experiments and matched controls are still feasible.

Additional measurements are required because local approximation errors can accumulate autoregressively:

- teacher-forced next-token loss / perplexity;
- sequence-length dependence;
- multi-step generation drift;
- span depth dependence;
- whole-model storage after repair.

Question: can multiple causal Transformer blocks be recompiled into a shorter task-distribution-conditioned map without unacceptable error accumulation?

### G7 — non-Transformer sequence control

Test at least one recurrent or state-space family (for example GRU/LSTM or a small state-space model).

This prevents a future positive result from being summarized merely as "residual feed-forward networks compress."

### G8 — arbitrary subgraphs and skip-graph cuts

Only after several architecture families have been mapped, relax the contiguous-block assumption.

Question: does simplification depend on clean sequential boundaries, or can it be defined on general learned computation graphs?

## Minimum confirmatory design per family

Each architecture/task family should use the same evidence discipline even when the mechanics differ.

1. **Eligibility gate** based only on baseline quality, fixed before simplification outcomes.
2. **Selection cohort** for any permitted family adaptation.
3. **Locked condition** before independent confirmation.
4. **Independent seed/model cohort** for the confirmatory result.
5. **Matched continued-training control** with the same repair budget.
6. **Seed/model-cluster inference**, not naive event-level confidence intervals.
7. Report both **functional fidelity** and **task utility**.
8. Report **core/subgraph reduction**, **whole-network reduction**, and **shell growth** separately.
9. Preserve negative and non-monotonic results.

For pretrained models, independent checkpoints or independently fine-tuned seeds should replace simple training-seed replication where appropriate.

## Adaptation must not become post-hoc rescue

The strongest failure mode in a generalization project is to keep changing the compiler until every architecture produces a positive result.

Therefore each family should have two phases:

### Discovery / adaptation cohort

A limited search budget may choose:

- boundary convention;
- candidate family from the allowed menu;
- repair locus;
- repair duration;
- quantization/storage scheme only after functional viability.

### Independent confirmation cohort

Freeze all of the above. No architecture-specific choice may be changed after confirmatory outcomes are inspected.

If confirmation fails, record the family as N or I and return to theory rather than silently retuning the same cohort.

## Candidate predictors of transfer

The generalization phase should try to predict **where** simplification works, not merely count successes.

Record candidate explanatory variables such as:

- original span depth and parameter count;
- residual vs non-residual topology;
- effective Jacobian / tangent rank of the removable computation;
- activation intrinsic dimension on the task distribution;
- replacement approximation error before repair;
- repairable shell effective dimension;
- composition subadditivity;
- error amplification across downstream layers;
- sequence/context length for recurrent/Transformer models;
- ratio of removed description length to added shell description length.

The objective is eventually to learn a condition of the form:

> simplification is likely when **task-effective computation complexity < architectural realization complexity**, and the residual mismatch lies within the bounded repair capacity of the surrounding system.

This is a hypothesis to test, not a conclusion to assume.

## Interpretation matrix

Several outcomes would all be scientifically useful.

### Broad Z results

If CNNs, ViTs, Transformer encoders, and small LMs simplify with essentially the same recipe, evidence for a genuinely general compiler becomes strong.

### Mostly A results

If the phenomenon recurs but requires architecture-specific grammars/boundaries, the project becomes a theory of **general simplification with family-specific compilation rules**.

### Mixed C/N results

If some families or tasks resist simplification, identify the distinguishing variables. A predictive boundary of applicability is more valuable than forcing a universal claim.

### Only current-family success

If external families repeatedly fail under fair adaptation budgets, the present result should be narrowed to a residual-CNN/task-manifold phenomenon. That is still a valid result.

## Priority order when experiments resume

1. **G0 portability control**
2. **G1 Fashion-MNIST residual CNN**
3. **G3 small ViT on a controlled image task**
4. **G4 CIFAR-10 small ResNet**
5. **G3b CIFAR-10 small ViT**
6. **G5 sequence Transformer encoder**
7. **G6 small decoder-only LM**
8. **G2 architecture panel / G7 recurrent control**
9. **G8 arbitrary subgraphs**

This ordering deliberately places a Transformer test early. Repeating many closely related CNN variants before testing a different computational family would provide less information about universality.

## What this roadmap does not require

- Every architecture does **not** need to simplify.
- Every architecture does **not** need the same candidate grammar.
- Every architecture does **not** need to reach the same byte count.
- Failure after a fair adaptation budget is not an implementation embarrassment; it is evidence about the applicability boundary.
- A positive result is not enough unless whole-network accounting and matched controls survive.

The intended output of this roadmap is therefore not a binary statement that "Canaria is universal." It is a **map of transfer regimes and the conditions that predict them**.
