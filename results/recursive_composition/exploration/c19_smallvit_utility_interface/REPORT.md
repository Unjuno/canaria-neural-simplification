# C19 exploratory report

C19 tested the self-anchored compressed hidden-interface mechanism on the established SmallViT central two-block replacement regime, where matched 4096-parameter replacements retain materially above-chance validation utility.

Fresh seeds were 1500–1502. Seed 1501 was retained but ineligible under the prospectively fixed teacher validation threshold (0.94444 < 0.95). Seeds 1500 and 1502 were eligible. No held-out test data was evaluated.

For both eligible seeds, final functional NMSE had the identical ordering:

`full_32 < anchored_16 < anchored_8 < frozen < sketch_only_16`.

Across the two eligible seeds, anchored-16 reduced final NMSE versus frozen by a mean 0.01332 and reached a mean final/full-32 ratio of 1.03983. Naive sketch-only-16 degraded final NMSE substantially (mean final/full-32 ratio 1.55008). Anchored-16 final validation accuracy averaged 0.85741, while the direct matched single averaged 0.84815; therefore this testbed did not exhibit C18's chance-level collapse.

Interpretation: C19 provides positive exploratory evidence that the self-anchor mechanism transfers to the SmallViT-family two-block regime while task utility remains materially above chance. It is not confirmatory evidence. The eligible exploratory cohort is only two seeds and used one fixed random basis.

A fresh confirmatory experiment should use fresh model seeds and fresh prospectively fixed basis choices, aggregate conservatively across basis choices, preserve exact learned-parameter matching, and keep held-out test outcomes out of all fitting/selection decisions.
