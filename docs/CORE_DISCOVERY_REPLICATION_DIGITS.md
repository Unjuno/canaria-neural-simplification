# Direct replication of compositional simplification on a residual MLP

This document records the fresh residual-MLP component-wise-versus-composed experiment. It is the strongest public matched-budget replication in the current repository.

## Question

> Under an explicit replacement grammar and validation fidelity/utility rule, does a fixed two-block span require fewer learned replacement parameters when its input-output map is fitted directly than when the two blocks are simplified separately?

This is an operational replacement-complexity question, not a Kolmogorov-complexity test.

## Locked design

Exploratory seeds: `1100–1103`.

Fresh confirmatory seeds: `1200–1207`.

Teacher:

- `sklearn.datasets.load_digits`;
- stem `64→64` + GELU;
- four residual MLP blocks;
- linear 10-class head.

Tested span: first two residual blocks.

Replacement grammar:

- **component-wise:** two bias-free residual bottleneck modules, each hidden width `h`;
- **composed:** one bias-free residual bottleneck module, hidden width `2h`.

At every budget point, learned replacement parameters are exactly matched:

```text
component-wise total = 2 × (64h + 64h) = 256h
composed total       =     (64·2h + 2h·64) = 256h
```

Budget grid:

`512, 1024, 1536, 2048, 3072, 4096, 6144` learned replacement parameters.

Compiler fitting:

- 600 updates per component module;
- 600 updates for the composed module;
- batch size 128;
- AdamW, lr `0.008`, weight decay `1e-5`.

Because each component module has half the learned parameters of the composed replacement, the two 600-update component fits match the composed fit in total parameter-update count and approximately in linear-layer multiply count.

## Selection and test isolation

Choose the smallest budget satisfying both:

1. validation span NMSE `<= 0.08`;
2. replacement validation accuracy within 2 absolute percentage points of teacher validation accuracy.

The final test set is not used for budget selection. The public runner evaluates test utility only after the minimum passing endpoint has been selected.

Primary endpoint:

> paired mean `log2(B_composed / B_componentwise)` across fresh seeds.

PASS requires the seed-bootstrap 95% CI upper bound below zero.

Secondary confirmatory endpoint:

> test-accuracy difference at validation-selected budgets.

PASS requires the paired bootstrap95 lower bound for `composed - component-wise` above `-0.02`.

Protocol SHA256:

`f0d16d813918f2a419a8ba1dfd0bf0efe663a0b7f8105288a84dc84f18530f5b`

## Fresh confirmatory result

**Overall: PASS.**

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

Composed selected the smaller minimum passing budget in **8/8** fresh seeds.

Primary:

- component-wise mean selected budget: **3584**;
- composed mean selected budget: **1728**;
- arithmetic mean budget reduction: **51.8%**;
- mean `log2(B_composed/B_componentwise)`: **-1.0519**;
- paired bootstrap95: **[-1.2075,-0.8962]**;
- geometric mean budget ratio: **0.4823×**;
- one-sided exact Wilcoxon `p=0.00390625`.

Secondary test utility:

- mean test-accuracy difference, composed minus component-wise: **+0.00583** (+0.583 percentage points);
- paired bootstrap95: **[+0.00306,+0.00806]**.

Thus, in this fresh cohort, the lower selected composed budgets did not achieve the primary result by accepting inferior held-out task utility.

## Mechanistic secondary — descriptive, not confirmatory causal proof

At fixed **2048 learned replacement parameters**, three fits were compared:

1. local component-wise fitting to separate intermediate targets;
2. the same two-module factorized topology jointly optimized end-to-end on the composed span target;
3. one composed module optimized on the span target.

Mean validation span NMSE:

- local component-wise: **0.1474**;
- jointly fitted two-module span: **0.0639**;
- single composed module: **0.0533**.

The joint span-objective condition recovers most of the local component-wise gap while preserving the two-module topology.

This is **consistent with** the composed functional objective/boundary accounting for a substantial part of the observed operational gap. However, this comparison had no confirmatory PASS/FAIL decision rule and does not uniquely identify a causal mechanism. The remaining difference between joint-factorized and one-module composed fits can reflect topology, optimization, inductive bias, or other factors.

## Supported interpretation

> In this residual-MLP task, fixed span, replacement grammar, optimization budget, and validation rule, the directly fitted composed span required substantially fewer learned replacement parameters than local component-wise simplification.

This experiment additionally provides descriptive evidence that changing the objective from local intermediate targets to the composed span target can recover much of the gap even when the two-module topology is retained.

## Not established

- universal mathematical/Kolmogorov subadditivity;
- task-universal or architecture-universal behavior;
- that the single-module topology is solely responsible;
- that the composed objective is the unique causal mechanism;
- that every span exhibits this effect;
- hardware/runtime benefit from the parameter reduction.

Machine-readable artifacts are under `results/core_discovery_digits/` and the public runner is `scripts/reproduce/core_discovery_digits/run_confirmatory.py`.
