# G6c Logit-aware compiler pilot protocol — v24

## Question
Can an output-aware, label-free compiler objective prevent the autoregressive rollout divergence observed in v23 natural-English character LM, without changing the 2-block/MLP24 compiler capacity?

## Frozen architecture/data
Reuse the v23 natural-English document split, character vocabulary, 4-block causal teacher, 2-block MLP24 compiler, sequence length 48, prompt length 12, rollout length 24, and baseline training procedure.

## Discovery seeds
3597 and 3598 only. These seeds are exploratory and excluded from confirmatory inference.

## Candidate objectives (fixed before discovery outcomes)
All use exactly the same compiler architecture and 512 unlabeled training windows. No ground-truth next-token labels enter compiler fitting.

- O0 `hidden_mse`: mean squared error between teacher and compiler residual-stream states.
- O1 `hidden_kl`: normalized hidden MSE + teacher-forced KL divergence between teacher and compiler next-token logits, temperature 1.0.
- O2 `hidden_kl_margin`: O1 + margin-weighted hard teacher-decision cross entropy. Positions where the teacher top-1/top-2 logit margin is small receive larger weight `1 + 2*exp(-margin)`.

Loss scaling is fixed as:
- hidden term = MSE / (mean teacher hidden variance + 1e-6)
- KL term coefficient = 1.0
- O2 decision term coefficient = 0.25

Compiler optimization: 20 epochs, AdamW lr=3e-3, weight_decay=1e-5, batch size 32.

## Selection rule
For each discovery seed measure tau=0 teacher-forced PPL utility and 24-token greedy rollout token agreement against the uncompiled teacher.

Select the objective with the highest mean rollout agreement across the two discovery seeds, subject to mean PPL utility >= 0.98. If objectives are within 0.01 rollout agreement, choose the simpler objective in order O0 < O1 < O2.

## Confirmatory gate after selection
Freeze the selected objective before any seed >=3600 is inspected. Confirmatory cohort is the first 8 seeds >=3600 passing the v23 baseline gate (validation PPL <=20 and token accuracy >=0.20).

Primary confirmatory conditions:
1. tau=0 compiled model: PPL utility lower 95% seed-bootstrap CI >=0.95 AND 24-token rollout agreement lower 95% CI >=0.90.
2. If tau=0 fails rollout, one preregistered bounded joint repair at tau=8 may be evaluated with the same v23 repair rule; adapted transfer requires both lower CIs >=0.95 (PPL utility) and >=0.90 (rollout agreement).

No q8/storage follow-up unless functional transfer passes.
