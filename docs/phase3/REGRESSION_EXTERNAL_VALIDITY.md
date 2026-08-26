# Phase 3 — regression task-type external validity

## Status

**Locked confirmatory result: `VALID_PASS`.**

This is post-v0.2.0 research on a separate branch/PR. It does not modify the frozen `v0.2.0-public-snapshot` claim boundary.

## Question

Does the operational component-wise-versus-composed replacement effect survive a genuine task-type change from supervised digits classification to tabular regression?

The testbed is `sklearn.datasets.load_diabetes`, using a four-block residual MLP with internal width 64 and a fixed first-two-block span.

## Evidence discipline

Exploration and confirmation were separated before confirmatory outcomes:

- exploration-only seeds: `2100–2102`;
- fresh confirmatory seeds: `2200–2207`;
- fixed train/validation/test split;
- feature and target normalization fit on the training split only;
- exact learned replacement-parameter matching at every budget;
- validation selects the minimum passing endpoint;
- test metrics are not selection variables;
- missing passing endpoints would be retained as censoring rather than dropped.

The confirmatory protocol was committed before the confirmatory workflow commit and before seeds `2200–2207` ran. See `results/phase3/regression_external_validity/CONFIRMATORY_PROTOCOL.json`.

## Exploration-only calibration

The provisional exploration rule used span NMSE `<= 0.08` and validation R² no worse than teacher minus `0.05`. It was too strict: neither topology reached a passing endpoint in any of the three exploration seeds.

The curves nevertheless showed a stable diagnostic pattern: composed span NMSE was lower than component-wise NMSE at every tested grid point in all three exploration seeds. By `h=24`, component-wise NMSE was approximately `0.104–0.115`, while composed replacements reached approximately the same fidelity at smaller widths.

Before any confirmatory seed was run, the final shared fidelity threshold was therefore locked at NMSE `<= 0.12`; the R² tolerance remained `0.05`. The budget grid was locked to `h=[2,4,6,8,12,16,20,24,32]`, where `h=20` adds resolution and `h=32` is a ceiling guard.

Exploration seeds are excluded from inferential counts and statistics.

## Locked replacement accounting

For internal width `d=64` and bottleneck width `h`:

- component-wise: two bias-free TinyRes replacements, each with `2*d*h` learned weights;
- composed: one bias-free TinyRes replacement with bottleneck width `2h` and `2*d*(2h)` learned weights.

Thus both conditions have exactly:

`256*h` learned replacement parameters

at every grid point.

The passing rule was locked as:

1. span NMSE `<= 0.12`;
2. replacement-network validation R² `>= teacher validation R² - 0.05`;
3. choose the minimum passing learned-parameter budget independently for each topology.

## Confirmatory result

All 8 fresh seeds produced both endpoints. No seed was dropped or censored.

| seed | component-wise budget | composed budget | log2(comp/sep) |
|---:|---:|---:|---:|
| 2200 | 5120 | 3072 | -0.73697 |
| 2201 | 8192 | 4096 | -1.00000 |
| 2202 | 5120 | 2048 | -1.32193 |
| 2203 | 6144 | 3072 | -1.00000 |
| 2204 | 8192 | 3072 | -1.41504 |
| 2205 | 5120 | 3072 | -0.73697 |
| 2206 | 6144 | 3072 | -1.00000 |
| 2207 | 6144 | 3072 | -1.00000 |

Summary:

- composed lower minimum passing budget: **8/8**;
- component-wise mean minimum passing budget: **6272**;
- composed mean minimum passing budget: **3072**;
- mean paired `log2(B_composed/B_componentwise)`: **-1.02636**;
- paired seed-bootstrap 95% CI: **[-1.18424, -0.86848]**;
- geometric mean budget ratio: **0.49095×**;
- one-sided exact sign test on non-tied seeds: **p = 0.00390625**.

The preregistered decision required all 8 endpoints, composed lower in at least 7/8, and a bootstrap 95% CI upper bound below zero. All three criteria passed, so the locked result is `VALID_PASS`.

## Utility boundary

Selected-endpoint test R² difference, composed minus component-wise, averaged only **+0.01657** across the 8 fresh seeds and was positive in 5/8 seeds. This is secondary and does **not** support a claim that composed replacement improves test utility.

The supported result is about the minimum **validation-passing replacement budget**, not superiority in downstream test performance.

Teacher performance also varied substantially across seeds (validation R² approximately `0.235–0.474`, test R² approximately `0.112–0.255`). No teacher-eligibility screen was preregistered, so every fresh seed remains in the result. This makes the result appropriately broad within this small testbed but also limits claims about strong-regressor regimes.

## Ceiling boundary

Seeds `2201` and `2204` reached their component-wise minimum passing endpoint at the maximum locked grid point, `h=32` / `8192` parameters.

These are **not censored**: a passing endpoint exists, and all lower locked budgets failed, so the minimum passing budget on the locked grid is defined. However, this indicates limited headroom on the component-wise side. The result should not be described as demonstrating a fully resolved asymptotic complexity curve beyond the tested grid.

For seed `2201`, the previous component-wise point `h=24` narrowly missed the NMSE threshold (`0.120387 > 0.12`) and `h=32` passed. For seed `2204`, `h=24` had NMSE `0.126973` and `h=32` passed.

## Isolation audit

The public runner:

- creates train/validation/test splits before fitting;
- computes feature and target normalization statistics from training indices only;
- trains replacement maps from training activations;
- computes span NMSE and task utility on validation data for endpoint selection;
- uses exactly matched learned parameter budgets at every grid point;
- computes a topology's test metric only after that topology has obtained its first validation-passing endpoint;
- never uses test metrics in the passing rule or budget selection.

This is a stronger selection-isolation design than the earlier SmallViT runner, which recorded test metrics for all candidates.

## Interpretation

This experiment materially extends the task-type boundary: the same operational compositional-simplification pattern was observed on a built-in tabular **regression** task, not only digits classification.

What it supports:

> Under this diabetes-regression task, residual-MLP span, replacement grammar, split, fidelity rule, and learned-parameter accounting, directly fitting the composed two-block span required a substantially smaller minimum validation-passing replacement budget than fitting the two implementation blocks separately.

What it does **not** establish:

- universal regression behavior;
- task-universal compositional simplification;
- grammar-independent or codec-independent complexity reduction;
- mathematical/Kolmogorov subadditivity;
- large-model or LLM external validity;
- general test-utility improvement;
- a fully resolved component-wise curve beyond the locked `8192`-parameter ceiling.

## Provenance

- tracking issue: #10;
- research branch: `research/phase3-regression-external-validity`;
- draft PR: #11;
- locked protocol commit: `0413d37969ea3bda0c03e2a79bdd7f2380b70c29`;
- confirmatory workflow commit: `c9715370703e6b8f33c6733fc51b2471edbb8c97`;
- confirmatory result commit: `a88ce25d5fdbef9a64b50d605ae5ed0971f6dc92`;
- confirmatory GitHub Actions run: `32978726716`.

## Stopping rule

The locked 8-seed confirmatory run is complete. Do not tune this protocol after the result. Any further regression dataset, stronger teacher, alternative grammar, or wider ceiling is a new experiment/protocol and must not replace this result.
