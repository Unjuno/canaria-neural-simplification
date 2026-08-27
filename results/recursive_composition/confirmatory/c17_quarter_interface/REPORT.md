# C17 — quarter-dimensional self-anchored teacher interface

Status: **CONFIRMATORY PASS**.

C17 tested whether a 16/64-dimensional teacher correction, with the remaining 48 hidden dimensions anchored to the pre-alignment Canaria hierarchy, can repair a recursive composition boundary across fresh model seeds and fresh basis choices.

## Locked cohort

- seeds: `1480–1487`; all retained;
- bases: identity first 16 coordinates plus three fresh random orthogonal 16D subspaces (`20261010–20261012`);
- exact learned replacement budget: 4096 parameters at each aggregate hierarchy level;
- final recursive fits use hierarchy outputs only;
- held-out test is evaluated only after fitting and validation metrics are complete.

## Confirmatory result

| Endpoint | Point | 95% paired-seed bootstrap | Gate | Result |
|---|---:|---:|---:|---|
| worst-basis minus frozen NMSE | -0.00699395 | [-0.00755723, -0.00651648] | upper < 0 | PASS |
| worst-basis / full-64 NMSE | 1.34146× | [1.30236, 1.38198] | upper < 1.50 | PASS |
| worst/best basis NMSE spread | 1.02246× | [1.01727, 1.02812] | upper < 1.15 | PASS |
| worst-basis validation accuracy − full-64 | -0.694 pp | [-1.157, -0.231] pp | lower > −3 pp | PASS |
| worst-basis test accuracy − full-64 | -0.972 pp | [-1.417, -0.528] pp | lower > −3 pp | PASS |

The worst tested quarter-interface basis improved the frozen hierarchy in **8/8** fresh seeds.

## Interpretation

Within this tested residual-MLP recursive hierarchy, full hidden-state supervision is not required for the established boundary-repair mechanism: a 16D teacher correction plus a 48D Canaria self-anchor was sufficient to produce a reproducible repair effect with bounded loss relative to full-64 alignment across the prospectively fixed basis family.

This does **not** establish a universal 16D interface, arbitrary/adversarial-subspace invariance, information-theoretic sufficiency, other architectures/tasks, or large-model behavior.
