# Recursive Canaria composition

Status: **C1–C21 completed; current recursive-composition line is at a release-review stopping point.**

This document is the current navigation surface for experiments on composing already-learned Canaria candidates into larger replacements.

## Core mechanism

The tested mechanism is not parameter averaging or algebraic merging. Adjacent learned Canaria candidates are composed into a cluster, selected cluster parameters are temporarily re-opened for boundary re-alignment, the cluster is frozen again, and a matched replacement is recursively compiled from the frozen cluster's input/output map.

In shorthand:

`local Canaria candidates -> joint boundary alignment -> freeze -> Canaria-to-Canaria recursive compile`

The main empirical failure mode is boundary/distribution mismatch. Strictly chaining recursively generated units without re-alignment accumulates substantially more functional error.

## Confirmatory evidence

| Stage | Scope | Status | Main supported result |
|---|---|---|---|
| C3 | residual MLP, single recursive recompilation | CONFIRMATORY PASS | jointly adapted cluster can be frozen and recursively recompiled with bounded penalty vs direct original-teacher compilation |
| C5 | SmallViT-family control | CONFIRMATORY PASS | recursive mechanism and joint-adaptation benefit reproduce outside the residual-MLP family, while the tested token-wise grammar remains an absolute-utility limitation |
| C7 | residual MLP, depth-2 hierarchy | CONFIRMATORY PASS | re-aligning the newly formed hierarchy boundary repairs strict depth-2 recursion and remains bounded vs direct compilation |
| C9 | residual MLP, depth-3 hierarchy | CONFIRMATORY PASS | repeated re-alignment at each hierarchy level controls depth-3 error; strict no-return recursion degrades strongly |
| C13 | residual MLP, 32/64-dimensional self-anchored interface | CONFIRMATORY PASS | half-dimensional teacher correction plus Canaria self-anchor repairs the recursive boundary with bounded loss vs full hidden alignment |
| C15 | residual MLP, fresh basis robustness | CONFIRMATORY PASS | C13 is not specific to one favorable 32D basis; worst tested basis remains bounded and improves frozen hierarchy |
| C17 | residual MLP, 16/64-dimensional self-anchored interface | CONFIRMATORY PASS | quarter-dimensional teacher correction plus a 48D Canaria self-anchor repaired the boundary across fresh model seeds and fresh bases; worst-basis/full-64 geometric NMSE ratio 1.3415× [1.3024, 1.3820] |
| C20 | SmallViT central two-block, 16/32-dimensional self-anchor | CONFIRMATORY PASS | worst-basis anchored-16 improved frozen recursion across 9 eligible fresh seeds; worst-basis/full-32 geometric NMSE ratio 1.0609× [1.0457, 1.0755], with validation/test safeguards passing |
| C21 | SmallViT central two-block, 8/32-dimensional self-anchor | CONFIRMATORY PASS | worst-basis anchored-8 improved frozen recursion across 11 eligible fresh seeds; worst-basis/full-32 geometric NMSE ratio 1.0962× [1.0838, 1.1089], with validation/test safeguards passing |

## Exploratory and negative chain

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
- **C18** — attempted self-anchor transfer across the full four-block SmallViT span. Relative ordering remained favorable (`full < anchored < frozen < naive sketch`), but the matched token-wise replacement grammar collapsed validation utility to chance (~0.10). This is a retained utility failure and is not evidence for full-model Transformer compression.
- **C19** — returned prospectively to the pre-existing C4/C5 central two-block regime rather than tuning the failed C18 grammar. In both eligible exploratory seeds, `full_32 < anchored_16 < anchored_8 < frozen < sketch_only_16`, while replacement accuracy remained materially above chance. This motivated C20 and C21.

## Current claim boundary

A communication-safe statement is:

> In the tested residual-MLP and SmallViT-family settings, already learned Canaria replacements can be recursively composed by temporarily re-opening newly formed composition boundaries for joint alignment, freezing them again, and compiling a new replacement from the frozen Canaria cluster. Strict recursive chaining accumulates substantially more error. The boundary-repair signal can be compressed when the unobserved complement is anchored to the existing Canaria hierarchy: this was confirmed for 32/64 and 16/64 teacher corrections in the residual-MLP hierarchy, and for 16/32 and 8/32 corrections on the SmallViT central two-block span across fresh model seeds and prospectively fixed identity/random basis families.

The SmallViT result is explicitly **span-scoped**. C18 shows that the current token-wise grammar does not preserve task utility when stretched across the full four-block SmallViT span, even though the relative self-anchor mechanism remains visible.

Do **not** convert this into claims of lossless composition, unlimited recursive depth, arbitrary-subspace invariance, universal minimum interface dimension, full-model Transformer compression, LLM-scale behavior, or teacher-free compilation.

## Evidence layout

- Confirmatory protocols/results/audits: `results/recursive_composition/confirmatory/`
- Exploratory and negative records: `results/recursive_composition/exploration/`
- Confirmatory runners: `scripts/recursive_composition/confirmatory/`
- Exploratory runners: `scripts/recursive_composition/exploration/`

The original per-experiment research branches and GitHub issues remain provenance records. This integrated tree is a review/navigation surface, not a rewrite of those histories.
