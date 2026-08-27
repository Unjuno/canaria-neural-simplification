# C15 basis robustness — confirmatory report

Decision: **CONFIRMATORY PASS**.

C15 tests whether the C13 half-interface self-anchor result survives fresh model seeds and fresh basis choices. The confirmatory endpoints use the **worst tested basis within each model seed**, rather than selecting a favorable basis.

## Cohort and bases

Fresh model seeds: `1460–1467` (8/8 retained).

Prospectively fixed 32D subspaces within every seed:
- `identity_first32`
- `random_20260920`
- `random_20260921`
- `random_20260922`

The random bases were not used in C13 or C14.

Held-out test evaluation occurred only after all fitting and validation metrics for all basis conditions and controls within a seed were complete.

## Locked endpoints

### P1 — worst tested basis repairs frozen hierarchy

- mean worst-basis ΔNMSE vs frozen = **-0.01342790**
- bootstrap95 = **[-0.01494948, -0.01185732]**
- all 8 model seeds negative
- **PASS**

### P2 — worst tested basis remains bounded vs full-64

- geometric mean worst-basis/full-64 NMSE ratio = **1.21762x**
- bootstrap95 = **[1.19167x, 1.24184x]**
- per-seed range = **1.15468x–1.26756x**
- locked upper bound = 1.30x
- **PASS**

### P3 — basis sensitivity is bounded

- geometric mean within-seed worst/best basis spread = **1.03531x**
- bootstrap95 = **[1.02512x, 1.04309x]**
- range = **1.00412x–1.04693x**
- locked upper bound = 1.15x
- **PASS**

### Validation utility guardrail

Using the lowest validation accuracy among the four bases within each seed:
- mean difference vs full-64 = **-0.556 pp**
- bootstrap95 = **[-1.065 pp, -0.093 pp]**
- locked lower bound > -2 pp
- **PASS**

### Held-out test safeguard

Using the lowest held-out test accuracy among the four bases within each seed:
- mean difference vs full-64 = **-0.500 pp**
- bootstrap95 = **[-0.667 pp, -0.333 pp]**
- locked lower bound > -2 pp
- **PASS**

## Interpretation

The 32-dimensional self-anchored teacher interface is not specific to the original C13 random basis in this testbed. Under a coordinate half-space plus three fresh random orthogonal half-spaces, even the worst tested basis reproducibly repaired the frozen recursive boundary and stayed within the prospectively fixed fidelity and utility margins relative to full hidden alignment.

This remains a scoped result. It does not establish arbitrary-subspace invariance, adversarial-basis robustness, a universal 32-dimensional interface, or large-model behavior.

## Reproducibility

- `PROTOCOL.json`
- `seed_rows.csv`
- `RESULT.json`
- `tools/audit_c15_basis_robustness.py`
