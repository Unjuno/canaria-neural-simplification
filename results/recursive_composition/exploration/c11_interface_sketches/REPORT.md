# C11 compressed hidden-interface sketches — exploratory report

Fresh seeds: `1420–1422`. Held-out test was not evaluated.

C11 tests whether a low-dimensional nested orthogonal sketch of the original 64-dimensional hidden interface is sufficient for top-boundary re-alignment before recursive recompilation.

## Aggregate validation results

| condition | hierarchy NMSE | final NMSE | final/full64 | final val acc |
|---|---:|---:|---:|---:|
| frozen | 0.06796 | 0.07137 | 1.424× | 0.9580 |
| sketch 4 | 0.23729 | 0.24066 | 4.814× | 0.9432 |
| sketch 8 | 0.17190 | 0.18019 | 3.639× | 0.9630 |
| sketch 16 | 0.11590 | 0.12601 | 2.533× | 0.9704 |
| sketch 32 | 0.07737 | 0.09239 | 1.852× | 0.9765 |
| **full 64** | **0.03250** | **0.05011** | **1.000×** | 0.9704 |
| direct original | — | 0.04335 | — | 0.9728 |

The final-NMSE order was identical in all three seeds:

`full_64 < frozen < sketch_32 < sketch_16 < sketch_8 < sketch_4`.

## Interpretation

The naive sketch hypothesis failed in this implementation. Matching only a projected subset of the hidden interface leaves the orthogonal complement unconstrained during joint adaptation. The optimizer can therefore improve the observed sketch while moving the unobserved interface into a representation that is harder to compose recursively.

This is consistent with C10: downstream/task-relevant agreement is not sufficient to define a stable recursive interface.

The result does **not** show that compressed hidden supervision is impossible. It points to a more specific next mechanism: use a low-dimensional teacher sketch for the observed directions while explicitly anchoring the unobserved directions to the pre-alignment Canaria hierarchy.

## Boundary

C11 is exploratory and uses one deterministic random orthogonal basis. No held-out test was evaluated.
