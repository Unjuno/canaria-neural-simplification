# G6d Trajectory-aware natural-text compiler results — v25

## Question
Can one bounded iteration of dataset aggregation on compiler-generated prefixes stabilize the free-running natural-English trajectory that failed under hidden-state MSE (v23) and teacher-forced logit KL (v24)?

## Discovery
Two excluded discovery seeds (3697/3698) compared equal-update compiler objectives:

| variant | mean PPL utility | mean 24-token rollout agreement |
|---|---:|---:|
| T0 — teacher-prefix O1 control | 0.99732 | 0.48828 |
| T1 — 8-step on-policy prefixes | 0.99691 | 0.50977 |
| T2 — 24-step on-policy prefixes | 0.99665 | **0.53125** |

T2 exceeded T1 by >0.01 and satisfied the PPL gate, so T2 was frozen before confirmatory seeds >=3700.

## Confirmatory cohort
Seeds 3700–3707 all passed the baseline gate and form the complete confirmatory cohort.

Architecture remained fixed:
- 4-block natural-English causal teacher;
- 2-block/MLP24 compiler;
- teacher 23,138 parameters;
- compiled 11,042 parameters;
- **52.2776% parameter reduction**.

The selected T2 objective used 15 teacher-prefix O1 warm-start epochs followed by one fixed prefix-pool aggregation over horizons 1..24 and 80 mixed refinement updates. No ground-truth next-token labels were used.

## Confirmatory result
- PPL utility: **0.99709**, 95% seed-bootstrap CI **[0.99627, 0.99784]**
- 24-token greedy rollout agreement: **0.61963**, CI **[0.51123, 0.72900]**
- exact 24-token continuation agreement: **0.32031**, CI **[0.18359, 0.50391]**
- mean first divergence position: **11.82** tokens, CI **[8.98, 15.06]**
- seed range of rollout agreement: **0.34245–0.88151**

The PPL criterion passes decisively; the rollout criterion fails decisively.

## Decision
**G6d / v25 = N — no transfer under tested budget.**

One-iteration trajectory-aware distillation is insufficient to make the 52.28%-smaller natural-English compiler reliably trajectory-stable across seeds.

A notable feature is high seed heterogeneity: some confirmatory models approach the 0.90 rollout target while others diverge strongly despite similarly high PPL utility. This suggests that the next useful question is not only stronger trajectory training, but also which teacher/compiler dynamical properties predict rollout stability.

Exploratory correlations across eight seeds are too underpowered for claims. Baseline validation/test PPL showed positive Pearson correlations (~0.69/~0.62) with rollout agreement, but n=8 and post-hoc analysis make this hypothesis-generating only.

## What is established across v23–v25
1. Hidden-state fidelity alone is insufficient.
2. Teacher-forced logit KL alone is insufficient.
3. A single fixed dataset-aggregation pass on compiler-generated prefixes is also insufficient for stable >=0.90 rollout agreement.
4. In all three phases, teacher-forced PPL can remain near-perfect while free-running trajectories diverge.

## Next discriminative options
- iterative dataset aggregation with prefix regeneration after each refinement stage, under a separately frozen budget;
- explicitly measure teacher margin/entropy and downstream trajectory amplification to predict which seeds are compressible;
- test a less aggressive compiler capacity reduction as a controlled capacity-vs-stability frontier rather than continuing objective tuning at one fixed 52.28% reduction.
