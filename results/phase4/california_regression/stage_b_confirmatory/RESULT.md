# Phase4 California Housing confirmatory result

Decision: **PHASE4_CONFIRMATORY_PASS**. Evidence class: **PROSPECTIVE_CONFIRMATORY**. All 8 fresh model seeds2900-2907 were retained.

## Locked gates

- Teacher held-out test R2 mean: **0.795060**, 95% CI **[0.7910913713697723, 0.7995540822785517]** — PASS versus lower95 > 0.70.
- Selected composed/component-wise budget geometric ratio: **0.476029**; mean log2 ratio 95% CI **[-1.328241785222183, -0.832518749639422]**; composed lower **8/8** — PASS.
- Selected held-out test R2 composed-minus-component-wise: **0.006858**, 95% CI **[0.004358858924945988, 0.00967018123690487]** — PASS versus -0.03 noninferiority margin.
- Technical duplicate-host science records and aggregate decision: **PASS** exact. Technical repeats do not increase scientific n.

## Scope

This is a larger-sample regression external-validity result on one fixed California Housing split using the same residual-MLP architectural family and first-two-block replacement grammar. It is stronger than the diabetes result with respect to teacher quality and dataset size, but it is not architecture-general, not eight independent datasets, and not a universal minimum-description or systems performance result. Selected budgets are minima only over the locked finite grid. No test metric selected the teacher recipe or replacement budget.
