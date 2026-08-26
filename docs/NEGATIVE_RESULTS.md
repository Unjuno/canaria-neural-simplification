# Negative results, rejected explanations, and invalidated evidence

Negative results are first-class evidence in Canaria. They constrain public claims and must not be hidden. Technically invalid experiments are kept separately as **invalidation history** and must not be treated as negative scientific evidence.

## Canary is not a necessary local condition

Strong simplification was frequently observed in low-Canary spans in the blinded confirmatory map. High Canary is therefore not a necessary local condition under the tested sensor definition.

## Canary is not a strong standalone predictor

Adding Canary percentile to width produced only a small LOSO AUC gain and the confidence interval crossed zero. Treat Canary as a partial sensor, not the root mechanism.

## Implementation boundaries are not always functional boundaries

Individual block replacement can fail while a wider merged span succeeds. Treating implementation blocks as privileged functional atoms is not supported.

## Plain 8-block CNN trainability

The original plain 8-block architecture collapsed to chance-level performance in pilot seeds. Confirmatory work moved to a residual 8-block network before simplification/Canary outcomes were inspected.

## Parameter count alone is incomplete

Equal-parameter interventions did not make repair behavior equivalent. Raw trainable parameter count is an incomplete proxy for task-effective repair dimension.

## Unlimited recursive recompilation is not supported

After an initial collapse and repair, further collapse to extremely small representations generally failed. Current evidence does not support monotonic compilation toward zero complexity.

## Low-bit and sparsity boundaries

- Naive global/tensor two-bit quantization failed badly; calibration and channel-wise scaling were required for competitive low-precision behavior.
- Sparse weights plus explicit indices can exceed dense low-bit storage; unstructured sparsity does not guarantee actual storage savings.
- Several sub-10-KB head designs preserved local transformations but failed whole-network utility.

## Autoregressive boundary

Across v22–v25, teacher-forced likelihood could remain close to the teacher while free-running trajectories diverged substantially. Tested natural-text post-hoc repair objectives did not close this rollout gap under the aggressive `4→2` setting.

## Factorized fitting alone does not explain staged gains

G17 showed that back-to-back `4→3→2` compiler fitting without task learning between stages was equivalent to direct `4→2` within the preregistered equivalence band. This rejects the explanation that the G15 staged result is merely “two smaller compiler fits are easier.”

## Matched normalized error is not matched task safety

G20e showed that after recontracting, a compiler could reach the same standardized functional-error target with fewer updates yet cause **more** immediate task NLL damage. Easier compiler optimization and downstream robustness are distinct.

## Hard task-damage veto can block the target architecture

G21's hard shadow-damage veto failed its target-reach criterion: only 10/12 fresh seeds reached the final two-block model, while mean compiler cost increased. The all-run PPL comparison is not capacity matched.

## One fixed risk cap did not solve the cost/utility trade-off

G27 exploratory tests found a trade-off rather than a Pareto improvement. No confirmatory G27 claim is made.

## Phase 2E is **not** negative evidence

Phase 2E must not be listed as evidence that stochastic 3-bit repair fails. It is `INVALIDATED_IMPLEMENTATION_BUG` because repair used raw `Xt` where the replacement was defined on internal activation `ta[0]`; equal width 64 hid the semantic error.

Consequences:

- the bugged `0/8` result is excluded from inference;
- the Phase 2I RNG explanation is retracted;
- 2H/2J interpretations that depended on 2E are weakened or confounded;
- corrected later work supports viability of short activation-domain repair in the tested residual-MLP family;
- Phase 2O remained `UNCERTAIN`, so a reliable composed repair-sample advantage is **not** established.

Preserved correction history:

- `results/phase2/precision_composition/CORRECTION_STATUS.json`
- `results/phase2/precision_composition/INVALIDATED_HISTORY.md`

## Interpretation rule

A failed intervention is evidence against that intervention under its protocol. An implementation-bugged experiment is not scientific negative evidence at all. Conversely, a successful small-model result is not evidence of universal simplifiability.
