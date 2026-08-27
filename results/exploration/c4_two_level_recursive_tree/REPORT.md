# C4 two-level recursive Canaria tree — exploratory report

Status: **EXPLORATORY COMPLETE**. No held-out test evaluation. Do not promote into the public claim set without fresh confirmation.

## Question

Does recursive Canaria-to-Canaria compilation remain controlled when applied twice rather than once?

C4 uses a deeper residual-MLP teacher and a matched-budget hierarchy:

```text
6 local Canarias (6 x width 4 = 3072 params)
        -> jointly adapted and frozen
3 pair Canarias (3 x width 8 = 3072 params)
        -> jointly adapted using adapted-cluster outputs only and frozen
1 final Canaria (width 24 = 3072 params)
```

The direct-original single control is also 3072 parameters.

After the six-candidate cluster is jointly adapted and frozen, **the original teacher span output is never used again as a recursive fit target**. Pair-local fits, pair-cluster joint adaptation, and the final single fit all use only adapted-cluster / pair-cluster behavior.

Fresh exploratory seeds: `1350, 1351, 1352`.

## Aggregate validation results

| representation | mean NMSE vs original teacher | mean validation accuracy |
|---|---:|---:|
| adapted six-candidate cluster | 0.06314 | 0.96296 |
| one-level recursive single `6->1` | 0.07162 | 0.96049 |
| pair cluster `6->3` | 0.07016 | 0.96173 |
| two-level recursive single `6->3->1` | 0.07700 | 0.96049 |
| direct original -> single | **0.05274** | **0.96420** |

## Recursive-depth penalty

Two-level / one-level original-teacher NMSE ratios:

- seed 1350: **1.0540x**
- seed 1351: **1.0901x**
- seed 1352: **1.0796x**
- geometric mean: **1.0745x**

Thus the second recursive level increased functional NMSE in all three exploratory seeds, but by only about 5–9% relative to one-level recursion under this schedule.

Two-level / direct-original NMSE ratios:

- `1.4731x`, `1.5158x`, `1.4010x`
- geometric mean: **1.4625x**

One-level / direct-original geometric mean ratio was **1.3612x**.

## Interpretation

C4 does not show lossless recursion. Every additional approximation level adds error, and direct original-teacher access remains best in functional NMSE.

However, the second recursive level did **not** produce catastrophic error compounding in these three seeds. The final two-level representation preserved the same mean validation accuracy as the one-level recursive representation (`0.96049`) while using the same 3072 learned-parameter representation budget at every hierarchy level.

The intermediate pair cluster is also informative: its mean NMSE vs the original teacher (`0.07016`) is slightly lower than the one-level single (`0.07162`). This suggests the `6 -> 3` recursive IR itself can remain coherent before the final `3 -> 1` contraction.

## Boundary

This is an exploratory three-seed result on one deeper residual-MLP digits setting. It supports a **controlled recursive-error-accumulation hypothesis**, not a general recursive compiler theorem. A fresh confirmatory cohort with a preregistered bound on the second-level penalty is required before a recursive-depth claim can be public-facing.
