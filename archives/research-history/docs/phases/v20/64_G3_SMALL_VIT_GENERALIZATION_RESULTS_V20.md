# G3 Small-ViT Generalization Results — v20

## Status
Architecture-family generalization test from the residual-CNN setting to a small Vision Transformer while holding the digits task fixed.

Confirmatory condition was locked before outcomes from seeds >=3200. Pilot seeds 3198/3199 are not included in confirmatory inference.

Protocol SHA256:
`8d668ffcd2a28214f1e7462ca79e54ded854290d3d4eaa9576c04dcc3e822043`

Run-script SHA256:
`0b778bbc5ae2f9a4bb0841a5af611f632a611e9177c37c1d7e21a8a681711053`

## Model and intervention
Teacher:
- 4 Transformer blocks, d=32, 4 heads, MLP hidden width 64.
- 35,306 parameters total.

Compiled model:
- entire 4-block core replaced by 2 Transformer blocks,
- d=32, 4 heads, MLP hidden width 32,
- compiler fit by activation MSE on 512 training examples without labels,
- compiled core frozen during shell repair.
- 14,058 parameters total.

Whole-model parameter reduction: **60.1824%**.

Confirmatory seed rule: first 8 seeds >=3200 with baseline held-out accuracy >=0.95.

Seeds used:
`3200, 3202, 3203, 3204, 3208, 3209, 3211, 3212`.

3201, 3205, 3206, 3207, and 3210 were baseline-ineligible. 3213/3214 were accidentally allowed to finish after the first 8 eligible seeds had already been determined and are **not used** in confirmatory inference.

## Primary functional results

| condition | mean utility | 95% seed-bootstrap CI | PASS95 seeds |
|---|---:|---:|---:|
| tau=0 | 0.9421 | [0.9321, 0.9514] | 2/8 |
| tau=2 | 0.9613 | [0.9548, 0.9676] | 7/8 |
| tau=8 | **0.9685** | **[0.9610, 0.9767]** | **7/8** |

Decision under preregistered rules:
- **Zero-shot transfer: FAIL**.
- **Adapted transfer at tau=8: PASS**.
- Transfer class: **A — adapted transfer**.

The strongest interpretation is not that the CNN compiler transfers unchanged. Instead, the simplification phenomenon transfers after an architecture-appropriate but preregistered adaptation: the Transformer core is replaced by a smaller Transformer core and the shell receives bounded repair.

## q8 whole-network follow-up
The protocol allowed a q8/zlib whole-network accounting follow-up only after primary adapted transfer passed.

Codec:
- each floating tensor quantized to signed symmetric int8 with one FP32 scale,
- explicit tensor-name and shape headers are included,
- resulting real byte stream compressed with zlib level 9,
- q8 weights are dequantized back into the model for accuracy measurement.

The byte counts are real serialized **parameter-state streams** for every tensor in the model. The Python/architecture decoder implementation is treated as shared code and is not charged. Therefore these numbers are appropriate for matched whole-network parameter-code comparison, but they are not standalone executable-package sizes analogous to the v19 exact 9,926-byte codec.

Results across the same 8 confirmatory seeds:

- mean matched-control q8+zlib size: **34,435.8 B**
- mean compiled q8+zlib size: **13,590.6 B**
- mean q8+zlib whole-network reduction: **60.53%**
- 95% CI of reduction: **[60.38%, 60.68%]**
- q8 matched-control utility: **0.9670**
- 95% CI: **[0.9580, 0.9766]**
- PASS95 seeds: **7/8**
- compiled q8 / compiled FP32 fidelity: **0.99875**
- 95% CI: **[0.99648, 1.00099]**

Thus the result is not only a parameter-count reduction. Under this explicit q8 real-byte parameter codec, the whole ViT parameter state is also approximately 60% smaller while maintaining matched-control-relative utility.

## What this establishes
This is positive evidence that task-conditioned computational simplification is **not restricted to the residual-CNN architecture family** used in earlier phases.

The experiment changes architecture family while holding task/data fixed, so it isolates architecture transfer better than changing architecture and dataset simultaneously.

## What this does not establish
- This is not yet dataset generalization: the task remains sklearn digits.
- It does not establish zero-shot compiler universality; zero-shot failed.
- It does not establish that every Transformer span is compressible.
- It does not test a language Transformer, autoregressive decoding, KV-cache behavior, or long-context dependence.
- Baseline eligibility used the held-out split as historically defined; future external-validity phases should preferably separate eligibility/validation from final test evaluation.
- The 60% reduction is architecture/code specific, not a universal compression ratio.
- The q8 byte stream assumes a shared architecture/decoder and should not be described as a fully standalone executable model file.

## Next discriminative experiments
1. **G3b: CIFAR-10 + small ViT** — task and input complexity shift within the Transformer family.
2. **G5: sequence Transformer encoder** — remove vision-specific patch structure.
3. **G6: small decoder-only language model** — test autoregressive Transformer computation.
4. Within small ViT, separately compile attention-only, MLP-only, and attention+MLP spans to test whether composition subadditivity generalizes mechanistically.
