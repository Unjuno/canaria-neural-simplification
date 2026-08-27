# C7 depth-two hierarchical Canaria recompilation — confirmatory report

Decision: **CONFIRMATORY PASS**.

C7 asks whether recursively generated pair-level Canaria units can themselves be composed, jointly adapted using only a lower-level Canaria IR target, and recompiled into a second-level single without substantial incremental loss relative to a flat single compiled from that same lower-level IR.

Fresh seeds: `1380–1389`. All 10 teachers passed the locked validation-accuracy eligibility threshold (`>=0.95`). All attempted seeds were retained. Test data was materialized only after every locked fit and validation metric for a seed was complete.

## Exact hierarchical budget

- Level 0: four `TinyRes(64,8)` units = **4096** aggregate learned parameters.
- Level 1: two recursively generated `TinyRes(64,16)` units (`C12`,`C34`) = **4096** aggregate learned parameters.
- Level 2: one `TinyRes(64,32)` (`C1234`) = **4096** learned parameters.

Thus every aggregate hierarchy level is exactly budget matched.

## Information isolation

The lower pair clusters may use their original pair-span targets during construction. They are then frozen and define the lower-level Canaria IR:

`IR1234(a0) = pair34(pair12(a0))`.

After this lower IR is defined, the original four-block output `a4` is not used as a target for any hierarchical adaptation or recompile. Level-2 joint adaptation targets only `IR1234`; the final depth-two single targets only the adapted `C12+C34` output. Original `a4` remains available only for validation measurement and the direct-original reference.

## Aggregate results

| condition | validation NMSE vs original | validation accuracy | test accuracy |
|---|---:|---:|---:|
| hierarchy, no Level-2 adaptation | 0.07262 | 0.96037 | 0.96844 |
| **depth-two hierarchy + Level-2 joint adaptation** | **0.06768** | **0.95963** | **0.97133** |
| flat single from same lower IR | **0.06633** | **0.95963** | **0.97178** |
| direct single from original teacher | 0.04276 | 0.96852 | 0.97867 |

The lower-level composed Canaria IR itself had mean validation NMSE `0.06226` versus the original teacher.

## P1 — Level-2 joint-adaptation effect

Per seed:

`D = NMSE(depth-two joint) - NMSE(depth-two no-adapt)`

- mean D = **-0.0049445**
- paired bootstrap95 = **[-0.0061376, -0.0037357]**
- all 10 seeds negative = **yes**
- locked rule: CI upper `< 0`
- **PASS**

The Level-2 joint adaptation step is therefore not cosmetic; it reproducibly removes error introduced by independently recompiled pair units before the final recompile.

## P2 — bounded incremental hierarchy cost

Per seed:

`R = NMSE(depth-two joint) / NMSE(flat lower-IR single)`

- geometric mean R = **1.02030x**
- paired bootstrap95 = **[1.01406x, 1.02707x]**
- per-seed range = **1.00750x–1.04139x**
- locked rule: CI upper `< 1.10x`
- **PASS**

Thus the extra recursive hierarchy level adds only a small bounded functional penalty relative to directly compiling a single from the same lower-level Canaria IR.

## P3 — validation task-utility guardrail

`depth-two joint - flat lower-IR single` validation accuracy:

- mean difference = approximately **0.0000** absolute accuracy
- paired bootstrap95 = **[-0.002222, +0.001852]**
- locked rule: CI lower `> -0.02`
- **PASS**

## Held-out test safeguard

`depth-two joint - flat lower-IR single` test accuracy:

- mean difference = **-0.000444** absolute accuracy (-0.044 percentage points)
- paired bootstrap95 = **[-0.001778, +0.000889]**
- locked rule: CI lower `> -0.02`
- **PASS**

## Direct-original reference

Direct original-teacher compilation remains materially better in functional NMSE. The geometric-mean depth-two/direct-original NMSE ratio is approximately **1.583x**, with per-seed values ranging from about **1.460x to 1.790x**.

C7 is therefore not a lossless-hierarchy result. Its narrower conclusion is that **the second recursive level itself adds little additional error once the lower-level Canaria IR already exists**. Most residual error is inherited from that lower IR.

## Supported claim

Within this residual-MLP digits testbed and exact matched replacement grammar, recursively generated pair-level Canaria units can be jointly adapted using a lower-level Canaria IR target and recompiled into a second-level single. The additional hierarchy cost is small and bounded relative to flat compilation from the same lower-level IR, and this holds on a fresh 10-seed confirmatory cohort.

## Boundaries

C7 does not establish:
- arbitrary-depth recursive hierarchy;
- lossless recompilation;
- cross-family depth-two behavior;
- high-utility large-model compression;
- LLM-scale behavior;
- that the original teacher can be discarded at arbitrary depth without accumulated lower-IR error.

## Reproducibility

- protocol: `results/confirmatory/c7_depth_two_hierarchy/PROTOCOL.json`
- runner: `scripts/confirmatory/c7_depth_two_hierarchy.py`
- per-seed rows: `results/confirmatory/c7_depth_two_hierarchy/seed_rows.csv`
- machine-readable result: `results/confirmatory/c7_depth_two_hierarchy/RESULT.json`
- independent endpoint audit: `tools/audit_c7_depth_two_hierarchy.py`
