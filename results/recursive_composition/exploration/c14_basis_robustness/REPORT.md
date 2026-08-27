# C14 basis robustness — exploratory report

C14 asks whether the C13 half-interface self-anchor result depends on one favorable basis.

Fresh model seeds: `1450–1452`. No held-out test evaluation.

Four prospectively fixed 32D subspaces were evaluated inside every model seed: one coordinate basis (`identity_first32`) and three independently generated random orthogonal bases (`20260910`, `20260911`, `20260912`).

## Result

All **12/12 model-seed × basis conditions** improved final validation NMSE relative to the frozen hierarchy.

Across all 12 conditions:
- anchored-32 minus frozen NMSE ranged from **-0.01676 to -0.01126**;
- anchored-32 / full-64 NMSE ranged from **1.09165x to 1.16442x**.

Within each model seed, the worst/best basis final-NMSE spread was:
- seed 1450: **1.02127x**
- seed 1451: **1.02416x**
- seed 1452: **1.05004x**

Identity-coordinate supervision behaved consistently with the independent random bases.

## Interpretation

The C13 effect does not appear to be an accident of its original random basis. In this exploratory cohort, changing the 32D subspace changed fidelity only modestly while preserving the qualitative repair effect in every tested combination.

This remains exploratory. It does not prove invariance to arbitrary subspaces, adversarial bases, or a universal sufficient interface dimension. A fresh confirmatory cohort with fresh random bases is required before promoting basis robustness to a public claim.
