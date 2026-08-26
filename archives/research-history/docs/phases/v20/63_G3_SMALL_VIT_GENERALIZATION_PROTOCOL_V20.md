# G3 Small-ViT Generalization Protocol — v20

Status: confirmatory condition lock before seeds >=3200 outcomes.

## Question
Does task-conditioned subnetwork simplification transfer from the residual CNN family to a Transformer/Vision-Transformer family when the task is held fixed?

This phase tests **phenomenon transfer**, not byte-for-byte transfer of the CNN compiler.

## Data
- `sklearn.datasets.load_digits`.
- Input normalized to `[0,1]`.
- Fixed stratified 70/30 split with `random_state=12345`.
- No augmentation.

Holding the task fixed isolates architecture-family shift.

## Teacher architecture
Small ViT:
- 8x8 grayscale image.
- patch size 2x2 -> 16 patch tokens.
- one class token.
- embedding width `d=32`.
- 4 pre-norm Transformer blocks.
- 4 attention heads.
- teacher MLP hidden width 64.
- final LayerNorm + 10-way linear classifier.
- dropout 0.

## Baseline training
- AdamW, lr `2e-3`, weight decay `1e-4`.
- 40 epochs, batch 128.
- baseline eligibility: clean held-out accuracy >= 0.95.
- seed rule: starting at 3200, take the first 8 eligible seeds. Eligibility depends on baseline only.

## Locked compiler condition
Pilot-only seeds 3198/3199 are excluded from confirmatory inference.

Replace the **entire 4-block Transformer core** by:
- 2 Transformer blocks,
- same `d=32`, 4 heads,
- compiler MLP hidden width 32.

The compiler is fit only to teacher residual-stream activations:
`h_embed -> h_after_block4`.
No labels or held-out accuracy are used to fit the replacement.

Calibration:
- first 512 training examples in deterministic dataset order.
- activation MSE objective.
- AdamW, lr `3e-3`, weight decay `1e-5`.
- 50 compiler-fit epochs.

Compiler parameters are frozen after fitting.

## Repair and matched control
Measure tau = 0, 2, 8.

Compiled repair trains only shell components:
- patch embedding,
- class token,
- positional embedding,
- final LayerNorm,
- classifier head.

The compiled 2-block core remains frozen.

Matched control starts from the same baseline model and receives the same additional classification-training epoch count with the original network intact.

Repair/continuation optimizer:
- AdamW lr `8e-4`, weight decay `1e-4`.

## Metrics
For each seed:
- baseline accuracy,
- matched-control accuracy at tau 2/8,
- compiled accuracy at tau 0/2/8,
- utility = compiled accuracy / matched-control accuracy (tau0 denominator = baseline accuracy),
- baseline and compiled parameter count,
- fixed-FP32 whole-model reduction.

The seed/network is the statistical unit.

## Decision rules
### Zero-shot transfer (Z)
PASS if at tau0:
- mean utility >= 0.95, and
- seed-bootstrap 95% CI lower bound >= 0.95.

### Adapted transfer (A)
PASS if at tau8:
- mean utility >= 0.95,
- seed-bootstrap 95% CI lower bound >= 0.95,
- mean whole-model fixed-FP32 reduction >= 0.40.

If Z fails but A passes, classify this architecture shift as `A: adapted transfer`.
If both fail under this locked adaptation budget, classify as `N` for the current adaptation budget; this does not prove impossibility under all Transformer-specific compilers.

## Secondary analyses
- tau2 utility.
- per-seed PASS95 count.
- whether failures correlate with baseline or matched-control accuracy.
- later, only if the primary adapted-transfer criterion passes, run a whole-network q8/zlib accounting follow-up without changing the functional condition.

## Interpretation guardrails
- A pass does not imply the same low-byte codec as the CNN experiments.
- A fail does not imply Transformer networks are intrinsically non-simplifiable; it rejects this preregistered adaptation budget.
- Canary is not a primary variable in this phase.
