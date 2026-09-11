# Phase3B stronger-teacher regression confirmation

Decision: **PHASE3B_CONFIRMATORY_UNCERTAIN**. All 8 fresh model seeds were retained.

- Teacher absolute test R2 mean: 0.232523, 95% CI [0.1716426637191226, 0.28459251713474154] — **UNCERTAIN** vs lower bound >0.20.
- Paired short25-minus-baseline60 teacher test R2: 0.026355, 95% CI [-0.024780231598993382, 0.07372409315015413] — **UNCERTAIN** vs lower bound >0.05.
- Composed/component-wise geometric selected-budget ratio: 0.482339; mean log2 ratio 95% CI [-1.1556390622295665, -1.0]; composed lower 8/8 — **PASS**.
- Selected test R2 composed-minus-component-wise: 0.011804, 95% CI [-0.013819829238314318, 0.03935874046838346] — **PASS** vs -0.05 margin.
- Technical duplicate host comparison: **PASS**.

Interpretation: the composition-budget effect and selected-endpoint utility safeguard pass under the stronger-teacher candidate protocol, but the prerequisite claim that short25 is materially stronger/useful on held-out test is not established. Therefore this experiment is not promoted as a stronger-teacher external-validity PASS. No seed, threshold, recipe, or historical Phase3 result is replaced.
