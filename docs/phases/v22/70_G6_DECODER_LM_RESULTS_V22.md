# G6 Small Decoder-Only Causal LM Results — v22

## Status
Confirmatory cross-architecture test of the previously fixed Transformer simplification recipe on a decoder-only causal next-token model.

Confirmatory protocol SHA256: `a780a26680987fcee2a2e6bc2d1f8247c65cea71fb0f8b9696c5c2566647c504`

Pilot seed 3399 is excluded from confirmatory inference. Confirmatory seeds: `3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407`.

## Controlled causal language
Each sequence is a deterministic prompted synthetic language of length 20. A 4-token prompt `[BOS, MODE, KEY, START]` determines the remaining 16 continuation tokens. Training loss and evaluation perplexity are computed only on continuation tokens. Train, validation, and test generators are separate fixed splits.

This is causal/autoregressive Transformer evidence, but **not natural-language evidence**.

## Architecture and intervention
Teacher: 4 causal Transformer blocks, d_model 24, 4 heads, MLP width 48, 20,800 parameters.

Compiled model: all 4 blocks replaced by 2 causal Transformer blocks, d_model 24, 4 heads, MLP width 24, activation-MSE fit on 512 unlabeled training sequences, compiled core frozen during repair, 8,704 parameters.

Parameter reduction: **58.15%**.

## Confirmatory functional results
Primary utility definitions:
- PPL utility = matched-control perplexity / compiled perplexity
- generation utility = compiled greedy continuation token accuracy / matched-control continuation token accuracy

| condition | PPL utility mean [95% CI] | generation utility mean [95% CI] | exact continuation rate | decision role |
|---|---|---|---|---|
| tau=0 | 0.9588 [0.9507, 0.9669] | 0.8359 [0.8029, 0.8699] | 0.8047 | zero-shot primary |
| tau=2 | 0.9736 [0.9663, 0.9801] | 0.8962 [0.8675, 0.9244] | 0.8774 | secondary |
| tau=8 | **0.9852 [0.9799, 0.9895]** | **0.9632 [0.9467, 0.9770]** | **0.9565** | adapted primary |

Preregistered thresholds:
- PPL utility lower 95% bound >= 0.95
- generation-token utility lower 95% bound >= 0.90

Decision:
- **Z — zero-shot transfer: FAIL**. Teacher-forced PPL narrowly passes, but autoregressive generation fails.
- **A — adapted transfer at tau=8: PASS**.

## Autoregressive drift is a distinct failure mode
At tau=0, teacher-forced continuation token accuracy remains 0.9873 and PPL utility lower95 is 0.9507. However generation-token utility lower95 is only 0.8029, with exact-sequence rate 0.8047.

Thus a compiler can look acceptable under teacher forcing while accumulating errors during free-running generation. After tau=8 shell repair, exact continuation rate rises to 0.9565 and the mean first-error position is 15.30 out of 16 continuation positions.

## q8 state-stream follow-up at tau=8
All floating tensors are quantized to signed symmetric int8 with one FP32 scale per tensor. Tensor names/shapes are encoded and the real parameter/state byte stream is compressed with zlib level 9.

- matched-control q8+zlib bytes: **20,844.9 B** mean
- compiled q8+zlib bytes: **8,780.4 B** mean
- q8+zlib reduction: **57.88%** [95% CI **57.78%, 57.97%**]
- q8 PPL utility: **0.9846 [0.9794, 0.9889]**
- q8 generation utility: **0.9590 [0.9433, 0.9725]**

The architecture/decoder program is shared and not charged, so these bytes are not a standalone executable/model-package size.

## Compute-history note
A 4-seed parallel confirmatory attempt produced no result files before timeout and was discarded without outcome inspection. In the q8 follow-up, evaluation batching was changed from 4x64 to 1x256 prompts only after exact equivalence was verified; no model, metric, threshold, seed, or codec changed.

## What this establishes
- The simplification phenomenon extends to a **causal decoder-only Transformer** in this controlled autoregressive task after bounded repair.
- Teacher-forced quality alone is insufficient to certify decoder simplification; generation drift can expose failures hidden by PPL/token accuracy.
- With tau=8 repair, a 58.15% parameter reduction and ~57.9% q8+zlib state-stream reduction coexist with high PPL and generation utility.

## What this does not establish
Natural-language transfer, pretrained LLM transfer, long-context robustness, KV-cache behavior, sampling robustness, arbitrary prompt distributions, standalone decoder/code MDL, or a universal compression ratio remain open.

## Next discriminative experiments
1. Natural-language or real-text decoder LM at small scale.
2. Generation-length sweep for error amplification.
3. Attention-only vs MLP-only vs full-block causal compilation.
4. Pretrained small language model with independent fine-tuning/checkpoint replicates.