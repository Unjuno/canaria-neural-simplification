# G6c Logit-aware natural-text compiler results — v24

## Question
Does adding teacher-forced logit KL to the v23 hidden-state compiler objective prevent the autoregressive rollout divergence seen on natural English?

## Discovery
Discovery seeds 3597/3598 compared three fixed, equal-capacity 2-block/MLP24 compiler objectives:

| objective | mean PPL utility | mean 24-token rollout agreement |
|---|---:|---:|
| O0 hidden MSE | 0.99748 | 0.75521 |
| O1 hidden MSE + logit KL | 0.99758 | **0.77799** |
| O2 O1 + margin-weighted teacher decision loss | 0.96214 | 0.74414 |

By the preregistered selection rule, O1 was frozen for confirmation. O2 failed the mean PPL utility >=0.98 gate.

## Confirmatory cohort
Seeds 3600–3607 all passed the pre-existing baseline gate and are the complete confirmatory cohort.

The compiler remains 2 causal blocks with MLP width 24:
- teacher parameters: 23,138
- compiled parameters: 11,042
- nominal parameter reduction: **52.2776%**

### tau=0 zero-shot
- PPL utility: **0.99668**, 95% seed-bootstrap CI **[0.99580, 0.99761]**
- 24-token greedy rollout agreement: **0.56169**, CI **[0.48681, 0.63737]**
- exact 24-token continuation agreement: 0.23047, CI [0.08594, 0.38281]
- mean first divergence position: 10.25 tokens, CI [7.82, 12.64]

PPL criterion passes; rollout criterion fails decisively. **Zero-shot transfer FAIL.**

### tau=8 bounded joint repair
- PPL utility: **0.94343**, CI **[0.93819, 0.94767]**
- rollout agreement: **0.42513**, CI **[0.36442, 0.48844]**
- exact agreement: 0.09375, CI [0.02344, 0.20313]

Both preregistered adapted criteria fail. **Adapted transfer FAIL.**

## Decision
**G6c / v24 = N — no transfer under tested budget.**

Teacher-forced logit KL does not solve the natural-text autoregressive trajectory problem under this compiler capacity and repair budget. The compiler preserves teacher-forced PPL almost perfectly while free-running greedy trajectories diverge substantially.

This result should not be interpreted as showing that logit KL is harmful relative to hidden-MSE. v23 and v24 confirmatory cohorts use different seeds. The supported statement is narrower: **the selected teacher-forced logit-aware objective is insufficient to cross the preregistered rollout threshold.**

## Storage follow-up
Not run. The protocol prohibited q8/storage optimization after functional failure.

## Next discriminative test
The failure specifically suggests a distribution-shift problem: the compiler is optimized on teacher/data prefixes but evaluated on prefixes produced by its own rollout. The next bounded adaptation should therefore train against teacher outputs on **compiler-generated prefixes** (one-step dataset aggregation / trajectory-aware distillation) while keeping compiler capacity fixed.
