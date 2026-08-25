# Direct cross-family replication of compositional simplification

## Purpose

This experiment closes the optional cross-family replication task by testing the **core phenomenon itself**, rather than another generic compression endpoint.

The question is:

> Under the same replacement grammar and task/fidelity criteria, can two learned Transformer blocks be represented more compactly when their composed input-output map is fitted directly than when the two blocks are simplified separately?

This is a direct follow-up to the original residual-CNN compositional-subadditivity evidence, using a Small Vision Transformer on sklearn digits.

## Architecture and fixed span

Teacher:

- 4 Transformer blocks;
- hidden dimension 32;
- 4 attention heads;
- MLP width 64;
- sklearn digits image classification.

The tested span is the **central two Transformer blocks, blocks 1 and 2**. This span was fixed before confirmatory outcomes to avoid embedding/head boundary effects.

## Replacement grammar

Each replacement operator is either:

- identity, with zero replacement parameters; or
- one Transformer block with hidden dimension 32, 4 heads, and MLP width in `{8,16,32,64}`.

Two strategies are compared.

### Component-wise

Block 1 and block 2 are fitted separately against their respective teacher intermediate targets, then composed at execution time.

### Composed

The input to block 1 is mapped directly to the output of block 2 by one replacement operator.

Each non-identity compiler receives the same 40 fit epochs. A component-wise candidate with two learned replacements therefore receives **twice as many compiler updates** as a one-operator composed candidate. The optimization budget does not favor the composed strategy.

## Pre-registered passing criterion

A candidate passes if both are true:

- training-held-out span NMSE `<= 0.12`;
- validation task utility `>= 0.95` relative to its baseline teacher.

Within each strategy, the selected representation is the passing candidate with the fewest replacement parameters; ties are broken by lower held-out NMSE.

Final test accuracy is not used for candidate selection.

Fresh confirmation uses the first 8 baseline-eligible seeds `>=9000`, with baseline eligibility defined only by validation accuracy `>=0.95`. Exploratory 8900-series seeds are excluded.

Protocol: `results/replication/vit_compositional/PROTOCOL_LOCK.json`.

## Confirmatory result

Eligible fresh seeds:

`9000, 9003, 9004, 9007, 9008, 9009, 9010, 9011`

In **all 8 seeds**, the minimum passing component-wise representation used two width-8 replacement blocks:

- replacement parameters: **9,808**;
- compiler updates: **640**.

The minimum passing composed representation used one width-8 or width-16 block:

- replacement parameters: **4,904–5,424**;
- compiler updates: **320**.

### Primary endpoint

Mean composed/component-wise complexity ratio:

**0.51988**

Geometric mean ratio:

**0.51926**

Paired seed-bootstrap 95% CI of the ratio:

**[0.50634, 0.53926]**

Composed representation smaller:

**8/8 fresh seeds**

Primary decision: **PASS**.

Equivalently, the selected composed replacement required about **48% fewer replacement parameters** than component-wise treatment under the locked passing criteria.

## Independent test utility

The final test set was not used to select candidates.

Selected composed representation:

- mean test utility: **0.97856**;
- bootstrap 95% CI: **[0.97090, 0.98562]**.

Selected component-wise representation:

- mean test utility: **0.98479**.

The composed representation is somewhat less faithful than component-wise fitting in normalized span error, as expected from using fewer parameters, but remains above the locked task-preservation criterion and generalizes to the untouched test split.

## Interpretation

This is direct cross-family evidence for **task-conditioned compositional simplification**:

> Two learned Transformer computations that require two replacement operators when treated separately can, under the same declared grammar and task/fidelity criteria, be represented by one substantially smaller operator when treated as a single composed function.

The result is stronger than a simple layer-count reduction because the comparison explicitly asks how much replacement complexity is needed under component-wise versus composed treatment.

It also does not rely on extra fitting effort for the composed model: the selected component-wise candidates used twice the compiler updates.

## Limits

This experiment does **not** establish:

- universal subadditivity for every Transformer span;
- mathematical or Kolmogorov complexity reduction;
- large-LLM generalization;
- that every composition admits a one-block replacement;
- codec-independent complexity;
- that the exact 0.52 ratio is universal.

The correct statement remains operational and grammar-dependent: under this task distribution, span, replacement family, and passing criteria, the composed learned function admitted a smaller task-preserving representation than component-wise treatment.

## Evidence files

- `results/replication/vit_compositional/PROTOCOL_LOCK.json`
- `results/replication/vit_compositional/confirm_summary.json`
- `results/replication/vit_compositional/seed_table.csv`
- `scripts/replication/vit_compositional.py`

## H / T / D / C / U

**H** — The minimum passing replacement complexity is lower for the directly composed two-block map than for separate component-wise replacement.

**T** — SmallViT, fixed central two-block span, common replacement grammar, locked NMSE/validation-utility thresholds, fresh first-8-eligible seed rule, test set isolated from selection.

**D** — **PASS**: composed was smaller in 8/8 seeds; bootstrap 95% upper bound of the complexity ratio was 0.539 < 1; mean composed test utility was 0.979.

**C** — The effect may depend on the task manifold, compiler grammar, span location, and Transformer size. The component-wise strategy also incurs intermediate-target distribution mismatch, which is part of the operational meaning of preserving implementation boundaries.

**U** — External validity beyond this small vision Transformer remains open.
