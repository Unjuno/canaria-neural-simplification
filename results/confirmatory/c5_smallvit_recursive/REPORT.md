# C5 cross-family recursive Canaria composition — confirmatory report

Decision: **CONFIRMATORY PASS**.

C5 tests whether the recursive mechanism previously confirmed in the residual-MLP digits testbed also appears in a SmallViT span under an exactly matched replacement grammar. C4 was exploratory; C5 used fresh seeds and locked eligibility, endpoints, margins, bootstrap settings, and test isolation before outcomes.

## Cohort

Attempted seeds: `1350–1361` (12 total).

Teacher eligibility was prospectively fixed at SmallViT validation accuracy `>= 0.95`. Eligible seeds were:

`1350, 1351, 1352, 1353, 1354, 1355, 1357, 1359, 1360`

Thus 9/12 seeds were eligible, exceeding the locked minimum of 8. Seeds `1356, 1358, 1361` were retained in the audit record as ineligible and were not replaced by rescue seeds.

## Exact matched grammar

Target span: central SmallViT blocks 1 and 2 (0-indexed).

Local cluster:
- two `TinyTokenRes(32,32)` modules;
- 2048 learned parameters each;
- cluster total = **4096 learned parameters**.

Recursive/direct single:
- one `TinyTokenRes(32,64)`;
- **4096 learned parameters**.

The cluster and single therefore have an exact learned-parameter budget match.

## Recursive procedure

1. Fit local candidates separately to the two teacher subspans.
2. For `all_unfrozen_recursive`, jointly adapt both candidates against the original two-block span target.
3. Freeze the adapted cluster.
4. Fit a fresh matched single using only `a0 -> frozen_cluster(a0)`.
5. In Stage 3, the original teacher span output is forbidden as a fit target.
6. Compare with both an unadapted-cluster recursive control and a matched single fitted directly to the original span.

Held-out test data was evaluated only after all locked fitting and validation metrics for a seed were complete.

## Aggregate eligible-seed results

| condition | validation NMSE vs original | validation accuracy | test accuracy |
|---|---:|---:|---:|
| all frozen -> recursive | 0.24062 | 0.77942 | 0.82058 |
| **all unfrozen -> recursive** | **0.21884** | **0.82510** | **0.85597** |
| direct original -> single | **0.20791** | **0.83128** | **0.86543** |

Eligible-teacher mean validation accuracy was 0.96132 and mean test accuracy was 0.96708. The restricted token-wise replacement grammar therefore remains materially below teacher task utility; C5 is not evidence of high-utility Transformer compression.

## Locked confirmatory endpoints

### P1 — joint adaptation improves recursive recompilation

Per eligible seed:

`D = NMSE(all_unfrozen_recursive) - NMSE(all_frozen_recursive)`

Results:
- mean D = **-0.0217802**
- paired bootstrap95 = **[-0.0269976, -0.0169880]**
- all 9 eligible seeds negative = **yes**
- locked rule: CI upper `< 0`
- **PASS**

### P2 — recursive penalty relative to direct recompilation is bounded

Per eligible seed:

`R = NMSE(all_unfrozen_recursive) / NMSE(direct_original_single)`

Results:
- geometric mean R = **1.05588x**
- paired bootstrap95 = **[1.04155x, 1.07081x]**
- per-seed range = **1.01331x–1.10316x**
- locked rule: CI upper `< 1.20x`
- **PASS**

This is a bounded-penalty result, not a lossless or equivalent-recompilation result.

### P3 — validation utility guardrail

`all_unfrozen_recursive - direct_original_single` validation accuracy:

- mean difference = **-0.006173** absolute accuracy (-0.617 percentage points)
- paired bootstrap95 = **[-0.014815, +0.003292]**
- locked rule: CI lower `> -0.03`
- **PASS**

### Held-out test safeguard

`all_unfrozen_recursive - direct_original_single` test accuracy:

- mean difference = **-0.009465** absolute accuracy (-0.947 percentage points)
- paired bootstrap95 = **[-0.019753, +0.002058]**
- locked rule: CI lower `> -0.03`
- **PASS**

## Supported interpretation

C5 provides fresh cross-family confirmatory evidence for the mechanism:

```text
local Canaria candidates
       C1 -> C2
          |
          | joint span-level adaptation
          v
    adapted cluster
          |
          | freeze
          v
  cluster as teacher / IR
          |
          | cluster-output-only fit
          v
          C12
```

The same qualitative mechanism previously confirmed in a residual MLP is therefore not specific to that architecture family in the tested settings. Jointly adapting the local candidate cluster before recursive recompilation improves functional fidelity relative to recursively recompiling the unadapted cluster, and the resulting recursive single remains within the prospectively fixed functional penalty bound relative to direct original-teacher recompilation.

## Boundaries

C5 does **not** establish:
- lossless recursive recompilation;
- arbitrary-depth recursive hierarchy;
- high-utility Transformer compression;
- architecture or task universality;
- large-model or LLM behavior;
- that the original teacher can be discarded at arbitrary recursive depth.

The next decisive question is whether recursively generated Canaria units can themselves be composed and recompiled at a second hierarchical level without uncontrolled error accumulation.

## Reproducibility

Protocol: `results/confirmatory/c5_smallvit_recursive/PROTOCOL.json`

Per-seed audit rows: `results/confirmatory/c5_smallvit_recursive/seed_rows.csv`

Machine-readable result: `results/confirmatory/c5_smallvit_recursive/RESULT.json`

Independent endpoint recomputation: `tools/audit_c5_smallvit_recursive.py`
