# C3 recursive Canaria recompilation — confirmatory report

Decision: **CONFIRMATORY PASS**.

Fresh seeds: `1330–1337`. All seeds retained. Protocol was committed before fresh outcomes. The runner was committed before execution and its Git blob SHA is `ebf01a6f22ef6557f9153e27e772ce94c3c0e809`.

## Question

Can an already-fitted cluster of adjacent Canaria candidates be jointly adapted to the wider span, frozen, and then used **by itself** as the teacher for a matched single Canaria, with reproducible improvement over weaker freeze schedules and a bounded fidelity penalty relative to direct access to the original teacher?

Three local `TinyRes(64,8)` candidates contain 3072 learned parameters total. The single recursively/directly compiled `TinyRes(64,24)` also contains exactly 3072 learned parameters.

Stage-3 recursive compilation uses only `a0 -> frozen_cluster(a0)`. It does not use the original teacher `a3` as its fit target.

## Aggregate validation results

| condition | cluster NMSE vs original | recursive single NMSE vs original | recursive val acc | test acc |
|---|---:|---:|---:|---:|
| all frozen -> recursive | 0.17297 | 0.17194 | 0.93981 | 0.94944 |
| edges only -> recursive | 0.06076 | 0.06488 | 0.96481 | 0.97333 |
| **all unfrozen -> recursive** | **0.04861** | **0.05859** | 0.96389 | **0.97667** |
| direct original -> single | — | **0.04487** | **0.96898** | **0.97889** |

## Locked confirmatory endpoints

### P1 — joint adaptation beats frozen recursive compilation

`D_frozen = NMSE(all_unfrozen_recursive) - NMSE(all_frozen_recursive)`

- mean: **-0.11334**
- paired seed-bootstrap95: **[-0.12734, -0.09991]**
- all 8 seeds negative: **yes**
- locked rule: CI upper `< 0`
- **PASS**

### P2 — recursive fidelity penalty is bounded relative to direct recompilation

`R = NMSE(all_unfrozen_recursive) / NMSE(direct_original_single)`

- geometric mean ratio: **1.30383x**
- paired seed-bootstrap95: **[1.27495x, 1.33991x]**
- per-seed range: **1.26196x–1.40887x**
- locked rule: CI upper `< 1.50x`
- **PASS**

This is explicitly a bounded-penalty result, not a lossless-recompilation result.

### P3 — full joint adaptation beats the preregistered edges-only control

`D_edges = NMSE(all_unfrozen_recursive) - NMSE(edges_only_recursive)`

- mean: **-0.006289**
- paired seed-bootstrap95: **[-0.008109, -0.004331]**
- all 8 seeds negative: **yes**
- locked rule: CI upper `< 0`
- **PASS**

### Required held-out task-utility safeguard

Test accuracy difference, `all_unfrozen_recursive - direct_original_single`:

- mean: **-0.002222** absolute accuracy (-0.222 percentage points)
- paired seed-bootstrap95: **[-0.005000, +0.000556]**
- locked safeguard: CI lower `> -0.02`
- **PASS**

Test was evaluated only after every locked model for a seed had already been fitted and validation metrics recorded. There was no condition or budget selection.

## Supported interpretation

Within this residual-MLP digits testbed and matched replacement grammar, the following recursive path is supported:

```text
local Canaria candidates
        C1 -> C2 -> C3
             |
             | jointly fit the candidate cluster to the wider original span
             v
       adapted cluster
             |
             | freeze the entire cluster
             v
       cluster as teacher / IR
             |
             | fit a fresh matched single Canaria using cluster outputs only
             v
            C123
```

The full-cluster adaptation step is not cosmetic: it improves recursive recompilation over both an unadapted cluster and the preregistered edges-only partial-adaptation control. The recursively generated single Canaria remains close in held-out task utility to a single Canaria trained directly from the original teacher.

## Important boundary

Direct access to the original teacher remains better in functional NMSE: recursive all-unfrozen mean NMSE is `0.05859`, direct-original mean NMSE is `0.04487`. C3 therefore does **not** establish lossless recursion.

C3 also does not establish:
- arbitrary-depth recursive Canaria trees;
- architecture/task universality;
- equivalence to a general-purpose compiler IR;
- large-model/LLM behavior;
- that the original teacher can always be discarded after one recursive level.

The new confirmatory statement should remain scoped to this tested three-candidate residual-MLP setting until further cross-family confirmation.
