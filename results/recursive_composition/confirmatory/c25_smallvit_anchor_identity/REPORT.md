# C25 confirmatory report

C25 tested whether the quarter-interface anchor-identity mechanism transfers confirmatorily to the utility-preserving SmallViT central two-block replacement regime.

The protocol was locked before outcomes. The committed runner (`1a75b138a40415ff401e12591bf38735f3941498`) and C4 dependency (`9cb59691b1f310d563c7e03cd49f39ee40d02e70`) were mirrored byte-identically into the local container and executed there. Environment: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0.

Fresh seeds were 1580–1591. Seeds 1580 and 1583 were retained but ineligible under the prospectively fixed teacher-validation threshold. Ten eligible seeds remained, exceeding the locked minimum of eight.

All six preregistered gates passed under the locked 100,000-resample paired bootstrap (RNG 20261330):

- Self-anchor vs frozen: mean ΔNMSE = -0.00752, 95% CI [-0.00916, -0.00595].
- Self-anchor vs naive sketch-only-8: mean ΔNMSE = -0.24044, CI [-0.25941, -0.22179].
- Self-anchor vs the best generic anchor selected within each seed: mean ΔNMSE = -0.38228, CI [-0.40819, -0.35562].
- Self-anchor/full-32 functional NMSE ratio: geometric mean 1.07090x, CI [1.05649x, 1.08362x], below the locked 1.30x bound.
- Validation accuracy difference vs full-32: mean -0.44 percentage points, CI [-1.74pp, +0.93pp], passing the -4pp safeguard.
- Held-out test accuracy difference vs full-32: mean -1.11pp, CI [-2.37pp, approximately 0pp], passing the -4pp safeguard.

Additionally, the self-anchor condition had lower final NMSE than frozen, sketch-only, and the per-seed best generic anchor in all 10 eligible seeds.

Interpretation: within this scoped SmallViT two-block regime, useful compressed-interface repair is specifically associated with preserving the sample-specific pre-alignment Canaria interface state. Generic complement constraints, distribution-matched shuffled states, zero anchors, and the sample-specific span input did not reproduce that effect. Together with residual-MLP C23, this provides confirmatory cross-architecture support for the anchor-identity mechanism.

This does not prove mathematical necessity or universal causality. It does not establish full-model Transformer compression, arbitrary architectures, LLM behavior, or teacher-free composition.
