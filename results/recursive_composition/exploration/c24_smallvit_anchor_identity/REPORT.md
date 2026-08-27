# C24 exploratory report

C24 tested whether the anchor-identity mechanism from residual-MLP C23 transfers to the established utility-preserving SmallViT central two-block regime with an 8/32-dimensional teacher correction.

The prospectively fixed runner (`b6e4e23dc5c9ae735d2a3ada9ae308944005b53e`) and C4 dependency (`9cb59691b1f310d563c7e03cd49f39ee40d02e70`) were mirrored byte-identically into the local container. The container environment was Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, and scikit-learn 1.8.0. Held-out test data was not evaluated.

All three fresh seeds 1570–1572 passed the prospectively fixed teacher validation eligibility threshold. All three produced the same final-functional-NMSE ordering:

`full_32 < anchor_self < frozen < sketch_only_8 < anchor_mean < anchor_shuffled < anchor_input < anchor_zero`.

Across the three seeds, `anchor_self` improved over frozen by mean NMSE difference -0.00751 and over naive sketch-only-8 by -0.24501. More importantly for the mechanism question, `anchor_self` beat the best generic anchor (best of input/mean/shuffled/zero within each seed) by mean NMSE difference -0.42670. Its geometric final-NMSE ratio to full-32 hidden alignment was 1.09193x. Mean validation-accuracy difference from full-32 was -1.60 percentage points.

Interpretation: this is positive exploratory evidence that the useful complement anchor is specifically associated with preserving the sample-specific pre-alignment Canaria interface state, and that the same qualitative separation seen in residual-MLP C23 transfers to this SmallViT two-block regime. Generic magnitude/marginal/sample-specific alternatives did not reproduce the effect.

This is exploratory only. Three seeds and one fixed projection are insufficient for a confirmatory cross-architecture mechanism claim. A next confirmatory experiment should use fresh model seeds, a fresh projection or prospectively fixed projection family, a conservative per-seed best-generic control, preregistered bootstrap margins, and held-out task safeguards.
