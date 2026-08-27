# C7 depth-2 hierarchical recursive Canaria recompilation — confirmatory report

Decision: **CONFIRMATORY PASS**.

C7 tests whether recursively generated Canaria units can be composed and recursively recompiled at a second hierarchy level on fresh seeds, and whether temporarily reopening the next hierarchy boundary for joint adaptation controls the error accumulated by strict frozen recursion.

Fresh seeds: `1380–1387`. All 8 retained. Protocol and runner were committed before outcomes. Held-out test data was evaluated only after every locked fit and validation metric for each seed was complete.

## Exact hierarchy budget

- Local level: four `TinyRes(64,8)` = **4096 parameters total**.
- Level 1: `C12` and `C34`, each `TinyRes(64,16)` = **4096 total**.
- Level 2 final: `C1234 = TinyRes(64,32)` = **4096 parameters**.

Thus learned replacement budget is exactly matched at each hierarchy level.

## Aggregate validation/test results

| condition | val NMSE vs original | val accuracy | test accuracy |
|---|---:|---:|---:|
| strict hierarchical frozen | 0.07312 | 0.95741 | 0.97250 |
| **hierarchical joint-adapt -> freeze -> recurse** | **0.05003** | **0.96435** | **0.97861** |
| single-level recursive | 0.05442 | 0.96667 | 0.97750 |
| direct original -> single | **0.04202** | **0.96667** | 0.97778 |

## P1 — Level-2 joint adaptation repairs strict depth-2 recursion

`D_frozen = NMSE(hierarchical_joint) - NMSE(hierarchical_frozen)`

- mean: **-0.0230943**
- paired bootstrap95: **[-0.0256055, -0.0210009]**
- all 8 seeds negative: **yes**
- locked rule: CI upper `< 0`
- **PASS**

## P2 — joint-adapted depth-2 penalty vs direct is bounded

`R_joint = NMSE(hierarchical_joint) / NMSE(direct_original_single)`

- geometric mean: **1.19038x**
- paired bootstrap95: **[1.15680x, 1.22551x]**
- locked rule: CI upper `< 1.40x`
- **PASS**

This is a bounded-penalty result, not equivalence or losslessness.

## P3 — joint-adapted depth-2 beats single-level recursion

`D_single = NMSE(hierarchical_joint) - NMSE(single_level_recursive)`

- mean: **-0.00438996**
- paired bootstrap95: **[-0.00557390, -0.00330829]**
- all 8 seeds negative: **yes**
- locked rule: CI upper `< 0`
- **PASS**

Thus the staged two-level hierarchy with re-alignment was not merely viable; in this cohort it was functionally better than jointly adapting all four local candidates in one recursive stage.

## P4 — strict no-return depth-2 recursion remains bounded

`R_strict = NMSE(hierarchical_frozen) / NMSE(direct_original_single)`

- geometric mean: **1.74066x**
- paired bootstrap95: **[1.67523x, 1.79590x]**
- locked rule: CI upper `< 2.25x`
- **PASS**

The strict hierarchy does accumulate a material penalty. This endpoint only establishes bounded accumulation for one additional recursive level in this testbed.

## Utility safeguards

Hierarchical-joint minus direct validation accuracy:
- mean: **-0.002315** (-0.231 percentage points)
- bootstrap95: **[-0.004167, -0.000463]**
- locked lower bound `> -0.02`
- **PASS**

Hierarchical-joint minus direct held-out test accuracy:
- mean: **+0.000833** (+0.083 percentage points)
- bootstrap95: **[-0.001111, +0.002500]**
- locked lower bound `> -0.02`
- **PASS**

## Supported interpretation

The confirmatory result supports the following scoped mechanism:

```text
local candidates
C1 C2 C3 C4
 |  |  |  |
 +--+  +--+
  |      |
joint pair alignment
  |      |
freeze + recursive compile
  v      v
 C12    C34
    \    /
    C12->C34
       |
       | temporarily unfreeze next hierarchy boundary
       | jointly align to wider original span
       v
 adapted hierarchy
       |
       | freeze
       | hierarchy-output-only recursive compile
       v
     C1234
```

In this residual-MLP testbed, recursively generated Canaria units can themselves be recursively composed at a second level with bounded error. Strictly freezing the hierarchy produces measurable accumulation, while reopening the next composition boundary for joint adaptation and freezing again substantially reduces that accumulation. This improvement was present in all 8 fresh seeds and also outperformed the single-level recursive baseline in all 8 seeds.

## Boundaries

C7 does **not** establish:
- unlimited recursive depth;
- lossless hierarchical compilation;
- architecture/task universality;
- that original-teacher access is never needed at deeper boundaries;
- LLM-scale behavior.

The next scientific question is no longer whether depth-2 recursion can work. It is how error scales with recursion depth and whether the re-alignment step yields a stable depth-error law.

## Reproducibility

- Protocol: `results/confirmatory/c7_depth2_recursive/PROTOCOL.json`
- Runner: `scripts/confirmatory/c7_depth2_recursive.py`
- Per-seed rows: `results/confirmatory/c7_depth2_recursive/seed_rows.csv`
- Machine-readable result: `results/confirmatory/c7_depth2_recursive/RESULT.json`
- Independent recomputation: `tools/audit_c7_depth2_recursive.py`
