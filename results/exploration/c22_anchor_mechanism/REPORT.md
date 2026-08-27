# C22 exploratory report — what must the complement anchor preserve?

C22 tested whether the self-anchor benefit comes from generic complement regularization or from preserving the sample-specific pre-alignment Canaria interface state.

The protocol and runner were locked before outcomes. Fresh seeds were `1550–1552`; no held-out test data was evaluated. Every aggregate hierarchy level used exactly 4096 learned parameters.

All anchored variants received the same 16/64-dimensional original-teacher correction. Only the orthogonal 48D complement anchor changed.

## Result

Mean final validation NMSE across the three seeds:

| condition | mean final NMSE |
|---|---:|
| full 64D teacher alignment | 0.05089 |
| **sample-specific pre-Canaria self-anchor** | **0.06468** |
| frozen hierarchy | 0.07316 |
| naive 16D sketch, complement unconstrained | 0.13774 |
| sample-specific span-input anchor | 0.74876 |
| mean Canaria-output anchor | 0.74758 |
| shuffled Canaria-output anchor | 0.77867 |
| zero anchor | 0.75917 |

`anchor_self` beat frozen, naive sketch, input, mean, shuffled, and zero anchors in **3/3 seeds**. Its mean NMSE was 0.887× frozen and 1.270× full-64 alignment.

The particularly informative control is `anchor_input`: it is sample-specific, but it is not the pre-alignment Canaria interface state. It failed almost as strongly as mean/zero anchors. Likewise, shuffling the pre-Canaria outputs preserves their marginal distribution while destroying sample pairing, and also failed strongly.

## Interpretation

This pattern argues against the explanation that self-anchor works merely because the unobserved complement is numerically constrained, sample-specific, or distribution-matched. In this testbed, the useful complement appears to be the **correct sample-specific recursive interface state carried by the pre-alignment Canaria hierarchy**.

This remains exploratory evidence. It does not prove mathematical necessity or causal sufficiency, and it does not establish the same mechanism for arbitrary architectures. A fresh confirmatory cohort should compare `anchor_self` against the best generic-anchor control prospectively, rather than selecting a favorable failed control after the fact.
