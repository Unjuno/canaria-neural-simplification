# G6c Logit-aware compiler confirmatory protocol — v24

## Frozen selection
The v24 discovery protocol fixed three label-free objectives on seeds 3597/3598. O1 (`hidden_mse + teacher-forced logit KL`) had the highest mean 24-token rollout agreement among candidates satisfying mean PPL utility >=0.98 and exceeded O0 by >0.01. O2 failed the PPL gate.

Selected objective: **O1**.

Pilot/discovery seeds 3597 and 3598 are excluded from confirmatory inference.

## Confirmatory cohort
First 8 seeds >=3600 satisfying the pre-existing v23 baseline eligibility gate:
- validation PPL <= 20.0
- validation token accuracy >= 0.20

No seed may be excluded based on compilation outcome.

## Model/intervention
Same v23 natural-English character dataset/document split and 4-block causal teacher.
Compiler is fixed at 2 causal blocks, MLP width 24, 52.2776% nominal parameter reduction.
Compiler fit uses 512 unlabeled training windows for 20 epochs with O1 only:
- normalized residual-stream MSE
- coefficient 1.0 teacher-forced KL between teacher and compiler logits
- no ground-truth next-token labels in compiler fitting.

## Primary zero-shot decision
At tau=0 require BOTH:
1. seed-bootstrap 95% lower CI of PPL utility >= 0.95;
2. seed-bootstrap 95% lower CI of 24-token greedy rollout agreement >= 0.90.

If either fails, zero-shot transfer FAILS.

## Preregistered bounded adapted follow-up
Independently of per-seed tau0 outcome, also evaluate the v23 bounded tau=8 joint repair:
- compiler + token embedding + positional parameter + final norm + LM head trainable;
- core remains the selected 2-block compiler architecture;
- 8 epochs, lr=3e-4;
- matched uncompiled control gets 8 epochs at lr=7e-4.

Adapted transfer requires BOTH:
1. lower 95% CI PPL utility >=0.95;
2. lower 95% CI rollout agreement >=0.90.

## Statistics
50,000 seed-cluster bootstrap resamples; seed/model is the statistical unit.

## Storage
No quantization/storage follow-up unless either zero-shot or adapted functional transfer passes.
