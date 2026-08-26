# G6d Trajectory-aware compiler confirmatory protocol — v25

## Frozen discovery selection
Discovery seeds 3697/3698 compared equal-update T0/T1/T2 objectives. All passed mean PPL utility >=0.98. Mean 24-token rollout agreement was T0=0.48828, T1=0.50977, T2=0.53125. T2 exceeded T1 by >0.01, so the tie rule did not apply.

Selected condition: **T2 — 15 O1 warm-start epochs + one fixed dataset-aggregation pass using compiler-generated prefixes over horizons 1..24 + 80 mixed refinement updates (40 data-prefix, 40 on-policy).**

## Confirmatory cohort
First 8 seeds >=3700 satisfying validation PPL <=20 and token accuracy >=0.20. Discovery seeds are excluded.

## Frozen model and optimization
- natural-English character LM/document split from v23;
- 4-block teacher -> 2-block/MLP24 compiler;
- 52.2776% parameter reduction;
- no ground-truth next-token labels in compiler fit;
- 320 total compiler optimizer updates, identical budget class to discovery;
- on-policy prefix pool generated once after warm-start and not regenerated.

## Primary decision
At tau=0, zero-shot functional transfer PASS requires BOTH:
1. 50,000-resample seed-bootstrap lower 95% CI PPL utility >=0.95;
2. lower 95% CI 24-token greedy rollout agreement >=0.90.

No additional task repair is permitted in v25. v25 isolates trajectory-aware compiler fitting after v23/v24 repair failures.

No q8/storage follow-up unless functional transfer passes.
