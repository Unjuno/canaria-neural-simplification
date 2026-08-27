# C23 confirmatory report — pre-Canaria interface state versus generic anchors

C23 prospectively tested the mechanism suggested by C22 on fresh seeds `1560–1567` and a fresh fixed 16/64 teacher-correction subspace. No held-out test data was evaluated; all fitting excluded the fixed validation split.

All five locked gates passed.

| endpoint | estimate | 95% CI | gate |
|---|---:|---:|---|
| self − frozen NMSE | -0.00713 | [-0.00785, -0.00652] | upper < 0 |
| self − sketch-only NMSE | -0.09012 | [-0.09707, -0.08368] | upper < 0 |
| self − best generic anchor NMSE | **-0.72066** | **[-0.73678, -0.70541]** | upper < 0 |
| self/full-64 geometric NMSE ratio | 1.29152× | [1.26622×, 1.32015×] | upper < 1.50× |
| self − full-64 validation accuracy | -0.42 pp | [-0.74 pp, 0.00 pp] | lower > -2 pp |

`anchor_self` beat frozen, naive sketch, and the per-seed best of `anchor_input`, `anchor_mean`, `anchor_shuffled`, and `anchor_zero` in **8/8 seeds**.

The P3 control is intentionally conservative: the generic anchor is chosen per seed after taking the minimum NMSE among all four generic alternatives, making the comparison maximally favorable to the generic-anchor explanation. The large separation remained.

## Interpretation

Within this residual-MLP recursive hierarchy, preserving the pre-alignment Canaria interface state is materially associated with successful compressed-interface repair. The effect is not reproduced by:

- simply constraining the complement to zero;
- preserving only the marginal distribution through shuffled pre-Canaria outputs;
- using a constant mean pre-Canaria state;
- using the sample-specific original span input instead of the pre-Canaria output; or
- leaving the complement unconstrained under projected MSE.

This supports a scoped mechanism interpretation: the unobserved complement appears to carry **state-specific interface information** needed by downstream recursive recompilation, rather than acting as a generic regularizer.

This result does not prove mathematical necessity or causal sufficiency, and it does not by itself establish the same anchor-identity mechanism in other architectures. Cross-architecture anchor-identity ablation is the next qualitative test.
