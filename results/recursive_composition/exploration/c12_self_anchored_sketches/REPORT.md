# C12 self-anchored compressed interface sketches — exploratory report

Fresh seeds: `1430–1432`. Held-out test was not evaluated.

C12 tests whether the C11 failure of partial hidden sketches is caused by unconstrained drift in unobserved interface directions. The teacher correction is observed only in a k-dimensional orthogonal sketch; the remaining directions are anchored to the pre-alignment Canaria hierarchy.

## Aggregate validation results

| condition | final NMSE | final/full64 | final/frozen | final val acc |
|---|---:|---:|---:|---:|
| frozen | 0.06986 | 1.450× | 1.000× | 0.9580 |
| sketch-only 16 | 0.16160 | 3.350× | 2.306× | 0.9593 |
| sketch-only 32 | 0.10995 | 2.280× | 1.571× | 0.9642 |
| anchored 8 | 0.06656 | 1.382× | 0.954× | 0.9617 |
| anchored 16 | 0.06300 | 1.307× | 0.902× | 0.9593 |
| **anchored 32** | **0.05792** | **1.202×** | **0.829×** | 0.9617 |
| full 64 | 0.04816 | 1.000× | 0.690× | 0.9667 |
| direct original | 0.04006 | — | — | 0.9704 |

The final-NMSE ordering was identical in all three seeds:

`full_64 < anchored_32 < anchored_16 < anchored_8 < frozen < sketch_only_32 < sketch_only_16`.

## Interpretation

The self-anchor mechanism directly fixes the failure mode suggested by C11. Low-dimensional teacher correction is useful when the unobserved orthogonal complement is prevented from drifting away from the existing Canaria interface.

`anchored_32` used teacher correction in only half of the hidden dimensions and reached a mean final/full64 ratio of **1.202x**. It beat the frozen hierarchy and the unanchored 32-dimensional sketch in every exploratory seed.

`anchored_16` also beat frozen in all three seeds, indicating that the effect is not restricted to the 32-dimensional condition.

This does not make the interface teacher-free: the unobserved directions are supplied by the pre-alignment Canaria hierarchy itself. The result instead supports a compressed-teacher-interface interpretation.

## Boundary

Exploratory only, one deterministic orthogonal basis, validation only.
