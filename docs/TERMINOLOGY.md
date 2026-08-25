# Current terminology

This glossary defines the terms used in the **current public interpretation** of Canaria. Historical documents may use some words more loosely; use this file when writing new documentation or discussing the public snapshot.

## Canaria

The research program studying task-conditioned computational simplification, compositional replacement, training-time consolidation, and related functional representations of learned neural computation.

`Canaria` is the project name. It should not be used as if it were one single fixed compression algorithm.

## Canary signal

A historical/local sensor associated with boundary stress or sensitivity in the original program.

Current interpretation:

- it can be informative in some settings;
- it is **not** a necessary local condition for simplification;
- it is **not** established as the causal mechanism;
- the current project should not be summarized as "find high-Canary regions and compress them."

## Component

A chosen implementation-level unit such as a block, layer, submodule, or bounded learned computation.

A component boundary is not assumed to be a natural functional boundary.

## Span

A contiguous learned computation between two chosen interfaces. A span may contain one or multiple implementation components.

The important object for replacement is the span's input-output function under the task/data distribution, not necessarily its internal implementation details.

## Boundary expansion

Moving the replacement boundary outward to include a wider computation when a narrower/local replacement fails or is unnecessarily complex.

Boundary expansion is one empirical reason implementation blocks should not be treated as privileged functional atoms.

## Replacement / compiler

A **replacement** is the smaller mechanism inserted in place of a learned span.

A **compiler** is the procedure that fits/selects/constructs such a replacement from calibration or training data under a declared candidate grammar and budget.

The term is intentionally broader than a conventional source-code compiler. It refers to translating an existing learned computation into another functional representation.

## Candidate grammar

The declared family of replacement mechanisms allowed in an experiment: for example affine maps, small neural modules, low-rank operators, structured sparse operators, or other bounded function classes.

Any description-complexity claim is conditional on the declared grammar/accounting unless explicitly shown otherwise.

## Functional transfer

Training/fitting a replacement to reproduce the relevant function of the source span before or during a structural consolidation event.

Functional transfer is a **handoff**, not necessarily a requirement for perfect teacher equivalence.

## Commit

The structural act of replacing the source span with the candidate replacement in the task model.

Pre-commit fit quality and post-commit task utility are distinct quantities.

## Recontracting

Continued task learning after a structural consolidation, during which the remaining model reorganizes around the new smaller mechanism.

Current evidence indicates that recontracting can:

- make the next compiler easier to optimize in normalized functional-error terms;
- simultaneously increase downstream sensitivity to residual approximation error.

Therefore recontracting should not be described merely as "damage repair."

## Compositional simplification

The central empirical phenomenon:

> several learned computations that are difficult or expensive to simplify component-wise can sometimes admit a smaller task-preserving representation when treated as one composed input-output function.

This is an operational/task-conditioned claim, not a theorem that mathematical function composition always reduces complexity.

## Composition subadditivity

An operational condition in which the measured replacement/description complexity of a composed span is lower than the corresponding component-wise complexity according to the declared experimental grammar/accounting.

The original confirmatory program observed this frequently, but the exact probability is setting/grammar dependent.

## Task-conditioned complexity

Complexity measured relative to:

- a task/data distribution;
- a declared fidelity/utility requirement;
- a candidate representation grammar;
- an accounting rule.

It is not automatically the same as global function complexity over the full mathematical input domain.

## Description complexity / description length

The size/cost assigned to a replacement or whole model under a declared representation/codec/accounting scheme.

Keep separate:

- parameter count;
- scalar support count;
- nominal bits;
- entropy estimates;
- real serialized bytes;
- compiler-update cost;
- FLOPs;
- wall-clock time;
- memory traffic;
- energy.

These are related but not interchangeable.

## Fidelity

How closely a compiled/replaced model reproduces a specified teacher/internal function or trajectory under a declared metric.

Examples include hidden-state error, logits, rollout agreement, or other functional comparisons.

Fidelity is not identical to task utility.

## Task utility

Performance on the actual task objective relative to an appropriate reference/control.

For language modeling this may include perplexity or another task metric. For autoregressive models, teacher-forced likelihood alone is not sufficient evidence of full trajectory fidelity.

## Matched continued-training control

An uncompiled/reference model that receives the same additional task-training budget/minibatch schedule as a compiled candidate when measuring recovery/recontracting.

This control prevents ordinary continued training from being misclassified as compilation recovery.

## Immediate task damage

The task-loss increase caused immediately after committing a candidate, usually measured relative to the corresponding teacher/reference state.

It can differ strongly from normalized internal compiler error.

## Downstream sensitivity

How strongly residual approximation error at a replacement boundary affects later computation or task loss.

Later Canaria experiments found that downstream sensitivity can increase after recontracting even while compiler fitting becomes easier.

## Remaining learning horizon

The amount of task learning available after a potential commit before the evaluation/end of the training schedule.

G18/G26 show that commit quality/future damage depends on this horizon; it should not be treated as irrelevant to structural decisions.

## Static / post-hoc simplification

A replacement performed after a trained computation has already formed, with zero or bounded adaptation afterward.

## Training-time consolidation

Structural replacement performed during learning, followed by continued task training and potentially later consolidation events.

This is the main dynamic extension of the original static compositional-simplification result.

## Compact functional representation / functional IR

A proposed systems-level representation that stores a task-conditioned functional form rather than necessarily storing all original parameter tensors.

This is currently an **application hypothesis**. Runtime storage, memory, latency, bandwidth, and energy benefits require direct systems measurements.
