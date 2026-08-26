# G6b rollout-horizon diagnostic results — v23

Exploratory follow-up on the same confirmatory seeds 3500-3507. This diagnostic does not change the preregistered G6b negative-transfer decision.

## Tau=0 compiled vs baseline greedy rollout agreement

| generated horizon | mean agreement | 95% seed-bootstrap CI |
|---:|---:|---:|
| 1 | **0.92188** | [0.89063, 0.95313] |
| 2 | **0.90039** | [0.86133, 0.93750] |
| 4 | **0.84082** | [0.78125, 0.89453] |
| 8 | **0.76514** | [0.69385, 0.83154] |
| 16 | **0.67529** | [0.59766, 0.75977] |
| 24 | **0.63265** | [0.55078, 0.72884] |

## Tau=8 joint-repair compiled vs matched control

| generated horizon | mean agreement | 95% seed-bootstrap CI |
|---:|---:|---:|
| 1 | **0.85547** | [0.81641, 0.89453] |
| 2 | **0.79688** | [0.76367, 0.83203] |
| 4 | **0.63965** | [0.57910, 0.70801] |
| 8 | **0.52832** | [0.43944, 0.62305] |
| 16 | **0.45166** | [0.37256, 0.53955] |
| 24 | **0.41130** | [0.33773, 0.48665] |

## Interpretation

The tau=0 compiler is locally close: first-character greedy predictions agree with the baseline about 92% of the time, and two-character prefixes about 90%. Agreement then decays monotonically with rollout horizon, reaching about 63% by 24 generated characters.

This supports an **error-amplification / trajectory-divergence** interpretation rather than a simple one-step prediction failure. The near-perfect teacher-forced PPL utility (0.997) and high one-step rollout agreement can coexist with poor long-horizon functional equivalence because a small early token difference changes all subsequent model inputs.

The selected tau=8 joint repair does not reduce this amplification; it starts from lower one-step agreement and remains worse at every measured horizon.

## Consequence for the next compiler objective

A residual-stream MSE objective can preserve average hidden-state / teacher-forced behavior without controlling the autoregressive transition induced by the model's own outputs. The next bounded adaptation should therefore test an objective that directly constrains output-distribution/logit divergence, preferably with a short rollout-aware or KL component, before moving to a pretrained model.

Because this is a post-confirmatory exploratory diagnostic, any new objective must be selected on pilot checkpoints and re-tested on fresh confirmatory seeds.
