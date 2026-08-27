# C9 depth-3 hierarchy with repeated boundary re-alignment — confirmatory report

Decision: **CONFIRMATORY PASS**.

C9 tests whether the depth-3 recursive hierarchy observed in exploratory C8 remains stable on fresh seeds when each newly formed hierarchy boundary is temporarily re-opened for joint alignment to the corresponding original span, then re-frozen before recursive recompilation.

Fresh seeds: `1400–1407`. All eight seeds were retained. Protocol and runner were committed before outcomes. Held-out test evaluation occurred only after every locked fit and validation metric for a seed had been completed.

## Constant representation budget

The learned replacement budget is exactly 4096 parameters at every hierarchy level:

- Level 0: 8 × `TinyRes(64,4)` = 4096
- Level 1: 4 × `TinyRes(64,8)` = 4096
- Level 2: 2 × `TinyRes(64,16)` = 4096
- Level 3: 1 × `TinyRes(64,32)` = 4096

## Aggregate results

| condition | validation NMSE | validation accuracy | test accuracy |
|---|---:|---:|---:|
| strict depth-3 | 0.21273 | 0.91806 | 0.93194 |
| **re-align each level** | **0.05197** | **0.96620** | **0.98000** |
| single-level recursive | 0.06364 | 0.96620 | 0.97750 |
| direct original single | **0.04454** | **0.97130** | **0.98306** |

Mean hierarchy trajectory:

- Level-1 full chain: `0.21209`
- strict Level-2 full chain: `0.21128`
- re-aligned Level-2 full chain: `0.07426`
- strict Level-3 final: `0.21273`
- re-aligned Level-3 final: `0.05197`

The strict path therefore accumulates substantial mismatch early and largely preserves it. Boundary re-alignment repairs a large fraction of that error at each subsequent hierarchy boundary.

## Locked endpoints

### P1 — repeated re-alignment repairs strict depth-3

`D_strict = NMSE(realign_each_level) - NMSE(strict_depth3)`

- mean: **-0.160762**
- paired bootstrap95: **[-0.175347, -0.147393]**
- all 8 seeds negative: **yes**
- rule: upper `< 0`
- **PASS**

### P2 — re-aligned depth-3 penalty vs direct is bounded

`R = NMSE(realign_each_level) / NMSE(direct_original_single)`

- geometric mean: **1.16729×**
- paired bootstrap95: **[1.14391×, 1.19063×]**
- rule: upper `< 1.40×`
- **PASS**

This is a bounded-penalty result, not equivalence or losslessness.

### P3 — re-aligned depth-3 beats single-level recursive

`D_single = NMSE(realign_each_level) - NMSE(single_level_recursive)`

- mean: **-0.011673**
- paired bootstrap95: **[-0.014813, -0.008807]**
- all 8 seeds negative: **yes**
- rule: upper `< 0`
- **PASS**

### P4 — Level-2 boundary re-alignment repairs the intermediate hierarchy

`D_level2 = NMSE(realigned Level-2 full chain) - NMSE(strict Level-2 full chain)`

- mean: **-0.137019**
- paired bootstrap95: **[-0.150018, -0.124628]**
- all 8 seeds negative: **yes**
- rule: upper `< 0`
- **PASS**

### Validation utility guardrail

Re-aligned depth-3 minus direct-original validation accuracy:

- mean: **-0.005093** (-0.509 percentage points)
- paired bootstrap95: **[-0.007870, -0.002315]**
- rule: lower `> -0.02`
- **PASS**

### Held-out test safeguard

Re-aligned depth-3 minus direct-original test accuracy:

- mean: **-0.003056** (-0.306 percentage points)
- paired bootstrap95: **[-0.004444, -0.001667]**
- rule: lower `> -0.02`
- **PASS**

## Supported interpretation

Within this 8-block residual-MLP digits testbed, recursively generated Canaria units can be composed through three hierarchy levels with bounded final error **when each newly formed hierarchy boundary is temporarily re-opened for joint alignment and then re-frozen before the next recursive compile**.

The strict no-return hierarchy is substantially worse. This supports boundary/distribution mismatch as a dominant failure mode in this tested hierarchy and supports repeated boundary re-alignment as an effective correction mechanism.

C9 does not establish unlimited recursion, losslessness, architecture universality, or large-model/LLM behavior.

## Reproducibility

- Protocol: `results/confirmatory/c9_depth3_realignment/PROTOCOL.json`
- Runner: `scripts/confirmatory/c9_depth3_realignment.py`
- Per-seed rows: `results/confirmatory/c9_depth3_realignment/seed_rows.csv`
- Result: `results/confirmatory/c9_depth3_realignment/RESULT.json`
- Independent endpoint recomputation: `tools/audit_c9_depth3_realignment.py`
