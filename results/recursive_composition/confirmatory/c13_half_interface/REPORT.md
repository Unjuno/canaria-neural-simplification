# C13 half-dimensional self-anchored interface — confirmatory report

Decision: **CONFIRMATORY PASS**.

C13 tests whether only half of a 64-dimensional hidden interface can be taken from the original teacher while the unobserved complement is anchored to the pre-alignment Canaria hierarchy.

## Cohort

Fresh seeds: `1440–1447` (8/8 retained).

The held-out test split was materialized only after all fitting and validation endpoints for each seed were complete.

## Conditions

- `frozen`: no top-boundary alignment.
- `sketch_only_32`: 32D projected hidden loss; complement unconstrained.
- `anchored_32`: 32D teacher correction, complementary 32D anchored to the pre-alignment Canaria hierarchy.
- `full_64`: full hidden alignment.
- direct matched single control.

Each aggregate hierarchy level contains exactly 4096 learned parameters.

## Aggregate validation NMSE

| condition | mean validation NMSE |
|---|---:|
| frozen | 0.070471 |
| naive sketch-32 | 0.102041 |
| **anchored-32** | **0.057640** |
| full-64 | 0.048267 |

## Locked endpoints

### P1 — anchored-32 repairs frozen hierarchy

- mean ΔNMSE = **-0.01283083**
- bootstrap95 = **[-0.01396431, -0.01169570]**
- 8/8 seeds improved
- **PASS**

### P2 — anchor repairs naive 32D sketch

- mean ΔNMSE = **-0.04440027**
- bootstrap95 = **[-0.04940808, -0.03958120]**
- 8/8 seeds improved
- **PASS**

### P3 — half-interface penalty vs full-64

- geometric NMSE ratio = **1.19368x**
- bootstrap95 = **[1.16267x, 1.22822x]**
- locked upper bound = 1.35x
- **PASS**

### Validation utility guardrail

- mean anchored-32 − full-64 = **-0.509 pp**
- bootstrap95 lower bound = **-0.741 pp** > -2 pp
- **PASS**

### Held-out test safeguard

- mean anchored-32 − full-64 = **-0.389 pp**
- bootstrap95 lower bound = **-0.583 pp** > -2 pp
- **PASS**

## Interpretation

In this recursive residual-MLP hierarchy, the teacher does not need to directly specify all 64 hidden dimensions during boundary repair. A 32-dimensional teacher correction combined with a complementary self-anchor from the existing Canaria hierarchy reproducibly improved both the frozen hierarchy and an unanchored 32D sketch, while remaining within a prospectively fixed fidelity margin relative to full hidden alignment.

This does **not** show that 32 dimensions is universally sufficient. C13 used one fixed orthogonal basis, one architecture family, and one task. Basis robustness is the next unresolved question.

## Reproducibility

- `PROTOCOL.json`
- `seed_rows.csv`
- `RESULT.json`
- `tools/audit_c13_half_interface.py`
