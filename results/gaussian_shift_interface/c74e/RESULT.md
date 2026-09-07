# C74E: final teacher-target blend

Evidence: PROSPECTIVE_EXPLORATORY. No reduced teacher-interface claim.

Decision: `HALF_BLEND_VALID_IMPROVEMENT_NOT_ESTABLISHED`.

| Stage | Mean gap to teacher (pp) | 95% CI (pp) | Reference gate |
|---|---:|---|---|
| B50 | -3.032407 | [-3.842591866850853, -2.268517389893532] | PASS |
| D64 | -3.009259 | [-3.8657404482364655, -2.152778208255768] | PASS |
| H64 | -3.078704 | [-4.004629701375961, -2.106481045484543] | PASS |
| S0 | -3.194445 | [-4.074074327945709, -2.361110597848892] | PASS |

B50-minus-S0 accuracy: 0.162037 pp, 95% [-0.2083323895931244, 0.5324069410562515], superiority gate UNCERTAIN.

## Descriptive contrasts

- B50_minus_D64: -0.023148 pp, 95% [-0.4398137331008911, 0.3703702241182327]
- D64_minus_S0: 0.185185 pp, 95% [-0.3472220152616501, 0.6944436579942703]
- S0_minus_H64: -0.115741 pp, 95% [-0.6018523126840591, 0.34722164273262024]

## Interpretation limits
The only final-fit intervention is target source: hierarchy-only S0, equal-weight hierarchy/teacher B50, or direct teacher D64. Architecture, initialization, minibatch-index stream, 384 calibration samples, 600 updates and batch128 are matched. H64 is also recorded.

B50 uses full teacher supervision in final compilation and is not evidence for a small feedback dimension. This experiment cannot show optimality of 50% or superiority over direct distillation unless directly supported by the reported contrast. No test-set use; seed intervals condition on fixed digits splits/subsets. Main unchanged. A numerical gate failure is not automatically proof of inferiority.
