# Direct replication of compositional simplification on a residual MLP

This document records the bounded closure experiment tracked in GitHub Issue #2.

## Why this experiment exists

The original Canaria discovery was that learned computation can be easier to replace when several learned components are treated as one composed input-output function rather than simplified independently at implementation boundaries.

To test whether that result was tied to the original architecture family, we ran a fresh confirmatory experiment on a different architecture: a four-block **residual MLP** trained on `sklearn.datasets.load_digits`.

This is an architecture-family replication. The task remains supervised classification, so it does not establish task-universal generality.

## Locked design

Exploratory seeds: `1100–1103`.

Fresh confirmatory seeds: `1200–1207`.

Teacher:

- input: 64 normalized digit pixels;
- stem: linear `64→64` + GELU;
- four residual MLP blocks;
- each teacher block: `x + Linear(128) → GELU → Linear(64)` applied to `LayerNorm(x)`;
- linear 10-class head.

The tested span is the **first two residual blocks**.

Replacement grammar:

- component-wise: two bias-free residual bottleneck modules, each hidden width `h`;
- composed-span: one bias-free residual bottleneck module with hidden width `2h`;
- total learned replacement parameters are therefore exactly matched: `256h` in both conditions.

Budget grid:

`512, 1024, 1536, 2048, 3072, 4096, 6144` learned replacement parameters.

Compiler optimization:

- 600 updates per component module;
- 600 updates for the composed module;
- batch size 128;
- AdamW, lr `0.008`, weight decay `1e-5`.

Because each component has half the learned parameters of the composed replacement, two component fits at 600 updates each have the same total parameter-update count and approximately the same linear-layer multiply count as one composed fit at 600 updates.

## Selection rule

The final test set is **not** used to select replacement budget.

For each condition, choose the smallest budget on the locked grid satisfying both:

1. validation span NMSE `≤ 0.08`;
2. replacement validation accuracy within 2 absolute percentage points of the teacher validation accuracy.

Primary endpoint:

> paired mean `log2(B_composed / B_componentwise)` across fresh seeds.

PASS requires the seed-bootstrap 95% CI upper bound to be below zero.

Secondary confirmatory endpoint:

> test-accuracy difference at the validation-selected budgets.

PASS requires the paired bootstrap 95% CI lower bound for `composed - componentwise` to be above `-0.02`.

Protocol SHA256:

`f0d16d813918f2a419a8ba1dfd0bf0efe663a0b7f8105288a84dc84f18530f5b`

## Fresh confirmatory result

**Overall: PASS.**

Selected minimum budgets by seed:

| seed | component-wise | composed |
|---:|---:|---:|
| 1200 | 3072 | 1536 |
| 1201 | 3072 | 2048 |
| 1202 | 4096 | 2048 |
| 1203 | 4096 | 1536 |
| 1204 | 3072 | 1536 |
| 1205 | 4096 | 2048 |
| 1206 | 3072 | 1536 |
| 1207 | 4096 | 1536 |

Composed-span used the smaller minimum budget in **8/8 fresh seeds**.

Primary result:

- mean component-wise selected budget: **3584 params**;
- mean composed selected budget: **1728 params**;
- arithmetic mean budget reduction: **51.8%**;
- mean `log2(B_composed/B_componentwise)`: **−1.0519**;
- paired bootstrap 95% CI: **[−1.2075, −0.8962]**;
- geometric mean budget ratio: **0.4823×**;
- one-sided exact Wilcoxon `p = 0.00390625`.

Secondary test-utility result:

- mean test-accuracy difference, composed minus component-wise: **+0.00583** (+0.583 percentage points);
- paired bootstrap 95% CI: **[+0.00306, +0.00806]**.

Thus the lower-budget composed replacements did not achieve the result by accepting worse held-out task utility in this confirmatory cohort.

## Mechanistic secondary control

At a fixed **2048-parameter** budget we also compared:

1. local component-wise fitting;
2. the same two-module factorized replacement architecture, but jointly optimized end-to-end on the composed span target;
3. one composed module.

Mean validation span NMSE across the fresh seeds:

- local component-wise: **0.1474**;
- jointly fit factorized span: **0.0639**;
- single composed module: **0.0533**.

The joint factorized control recovers most of the gap between local component-wise fitting and the single composed module. This is important: it suggests that much of the observed advantage comes from **treating the two blocks as one functional boundary/objective**, not merely from changing the topology from two small modules to one wider module.

The single composed module still showed a smaller additional NMSE advantage over the jointly optimized factorized control, but that comparison was a preregistered descriptive/mechanistic secondary, not the primary discovery test.

## Interpretation

This experiment supports an operational form of compositional simplification on a residual-MLP family:

> under a declared replacement grammar, optimization budget, validation fidelity criterion, and task-utility criterion, the composed two-block input-output function required substantially fewer learned replacement parameters than independent component-wise simplification.

It does **not** establish:

- universal mathematical or Kolmogorov complexity reduction under function composition;
- task-universal generality;
- that one-module topology is solely responsible for the effect;
- that every span or architecture exhibits the phenomenon.

The strongest new conclusion is that the core Canaria observation is not confined to the original implementation family, and that the relevant boundary is at least partly **functional/objective-defined rather than implementation-block-defined**.

Machine-readable artifacts are under `results/core_discovery_digits/`.
