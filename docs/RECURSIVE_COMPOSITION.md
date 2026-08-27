# Recursive Canaria composition

Status: **active research line; C1–C16 completed, C17 confirmatory execution in progress**.

This document is the current navigation surface for experiments on composing already-learned Canaria candidates into larger replacements.

## Core mechanism

The tested mechanism is not parameter averaging or algebraic merging. Adjacent learned Canaria candidates are composed into a cluster, selected cluster parameters are temporarily re-opened for boundary re-alignment, the cluster is frozen again, and a matched replacement is recursively compiled from the frozen cluster's input/output map.

In shorthand:

`local Canaria candidates -> joint boundary alignment -> freeze -> Canaria-to-Canaria recursive compile`

The important empirical failure mode is boundary/distribution mismatch. Strictly chaining recursively generated units without re-alignment accumulates substantially more functional error.

## Confirmatory evidence

| Stage | Scope | Status | Main supported result |
|---|---|---|---|
| C3 | residual MLP, single recursive recompilation | CONFIRMATORY PASS | jointly adapted cluster can be frozen and recursively recompiled with bounded penalty vs direct original-teacher compilation |
| C5 | SmallViT-family control | CONFIRMATORY PASS | recursive mechanism and joint-adaptation benefit reproduce outside the residual-MLP family, while the tested token-wise grammar remains an absolute-utility limitation |
| C7 | residual MLP, depth-2 hierarchy | CONFIRMATORY PASS | re-aligning the newly formed hierarchy boundary repairs strict depth-2 recursion and remains bounded vs direct compilation |
| C9 | residual MLP, depth-3 hierarchy | CONFIRMATORY PASS | repeated re-alignment at each hierarchy level controls depth-3 error; strict no-return recursion degrades strongly |
| C13 | 32/64-dimensional self-anchored interface | CONFIRMATORY PASS | half-dimensional teacher correction plus Canaria self-anchor repairs the recursive boundary with bounded loss vs full hidden alignment |
| C15 | fresh model seeds and fresh basis family | CONFIRMATORY PASS | C13 is not specific to one favorable 32D basis; worst tested basis remains bounded and improves frozen hierarchy |
| C17 | 16/64-dimensional self-anchored interface | ACTIVE | protocol and runner are locked; formal bootstrap decision and durable result are not yet recorded |

## Exploratory chain

- **C1** — first direct test of `Canaria cluster -> joint adaptation -> freeze -> single Canaria`; positive but not lossless.
- **C2** — freeze-boundary mapping; broad joint adaptation dominated narrow partial schedules.
- **C4** — SmallViT exploratory transfer; mechanism reproduced, replacement grammar limited absolute task utility.
- **C6** — first depth-2 hierarchy; strict recursion accumulated error, top-level re-alignment repaired much of it.
- **C8** — depth-3 exploration; repeated boundary re-alignment sharply separated from strict recursion.
- **C10** — supervision ablation; labels/logits can preserve task utility while failing to preserve a recursively compilable hidden interface.
- **C11** — naive low-dimensional hidden sketches; negative/informative result: unobserved complement drift caused degradation.
- **C12** — self-anchored sketches; anchoring the unobserved complement to the pre-alignment Canaria hierarchy repaired the C11 failure mode.
- **C14** — basis-robustness exploration for the 32D self-anchored interface.
- **C16** — teacher-correction dimension frontier over 8/16/24/32 dimensions; all tested dimensions repaired frozen hierarchy in the exploratory cohort, with fidelity improving smoothly as dimension increased.

## Current claim boundary

A communication-safe statement is:

> In the tested residual-MLP and SmallViT-family settings, already learned Canaria replacements can be recursively composed by temporarily re-opening newly formed composition boundaries for joint alignment, freezing them again, and compiling a new replacement from the frozen Canaria cluster. Strict recursive chaining accumulates substantially more error. In the residual-MLP hierarchy, the boundary-repair signal can also be compressed: a 32/64-dimensional teacher correction plus a complementary Canaria self-anchor remained effective across fresh model seeds and fresh coordinate/random basis choices.

Do **not** convert this into claims of lossless composition, unlimited recursive depth, arbitrary-subspace invariance, universal minimum interface dimension, LLM-scale behavior, or teacher-free compilation.

## Evidence layout

- Confirmatory protocols/results/audits: `results/recursive_composition/confirmatory/`
- Exploratory records: `results/recursive_composition/exploration/`
- Confirmatory runners: `scripts/recursive_composition/confirmatory/`
- Exploratory runners: `scripts/recursive_composition/exploration/`

The original per-experiment research branches and GitHub issues remain provenance records. This integrated tree is a review/navigation surface, not a rewrite of those histories.
