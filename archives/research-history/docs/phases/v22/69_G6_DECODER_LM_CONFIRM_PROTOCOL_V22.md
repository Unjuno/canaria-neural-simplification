# G6 Small Decoder-Only Causal LM Confirmatory Protocol — v22

## Lock point
Written after pilot seed 3399 and before inspecting any outcome from seed >=3400. Pilot seed 3399 is discovery-only and excluded from confirmatory inference.

## Scientific question
Does the Transformer simplification recipe already fixed before G5 transfer to a causal decoder-only next-token model, and can it preserve not only teacher-forced perplexity but also bounded autoregressive rollout quality?

## Data and evaluation separation
Synthetic prompted causal language from the v22 pilot protocol.
- train split: fixed generator seed 20260824
- validation split: fixed generator seed 20260825
- test split: fixed generator seed 20260826

Baseline eligibility uses validation continuation-token accuracy only. Confirmatory effect metrics use the disjoint test split.

## Architecture and compiler — frozen from pilot
Teacher: depth 4 causal Transformer blocks, d_model 24, 4 heads, MLP width 48.

Compiler:
- replace all 4 blocks by 2 causal Transformer blocks
- d_model 24, 4 heads, MLP width 24
- fit by residual-stream MSE on 512 unlabeled training sequences
- 50 compiler-fit epochs
- no next-token labels/loss in compiler fit
- compiler frozen during repair

No candidate-family, width, depth, calibration-size, or compiler-epoch search is permitted on the confirmatory cohort.

## Seed queue and eligibility
Scan seeds from 3400 upward in order. Eligible iff baseline validation continuation token accuracy >=0.95. Take the first 8 eligible seeds. No confirmatory-outcome exclusions.

## Conditions
- tau=0: no task repair; zero-shot transfer
- tau=2: secondary repair condition
- tau=8: adapted-transfer condition

For tau>0, only shell parameters are trainable: token embedding, position embedding, final LayerNorm, LM head. The compiled 2-block core is frozen. Matched controls continue training the uncompiled teacher for the same task epochs and optimizer budget.

## Metrics
Teacher-forced: test continuation NLL, perplexity, continuation token accuracy, `ppl_utility = control_ppl / compiled_ppl`.

Autoregressive greedy rollout from the 4-token prompt: continuation token accuracy, exact continuation sequence-match rate, mean first-error position, `generation_token_utility = compiled_gen_token_acc / control_gen_token_acc`.

## Confirmatory decision rules
Seed is the resampling/cluster unit. Use 50,000 seed bootstrap draws.

### Z — zero-shot transfer PASS
Both must hold at tau=0:
1. lower 95% bootstrap bound of mean `ppl_utility` >= 0.95
2. lower 95% bootstrap bound of mean `generation_token_utility` >= 0.90

### A — adapted transfer PASS
If Z fails, tau=8 is classified A only if both hold:
1. lower 95% bootstrap bound of mean `ppl_utility` >= 0.95
2. lower 95% bootstrap bound of mean `generation_token_utility` >= 0.90

Tau=2 is reported but does not replace the preregistered tau=8 adapted decision. Exact-sequence match and first-error position are mandatory secondary diagnostics.

## Whole-network state-stream follow-up
Only if Z or A passes:
- q8 signed symmetric per floating tensor with one FP32 scale
- explicit tensor names/shapes in the state stream
- zlib level 9
- real parameter/state-stream bytes, not a standalone executable/model-package size
- report q8 PPL utility and generation-token utility

## Interpretation constraints
A positive result does not establish natural-language generalization, large-LLM transfer, long-context robustness, or universal compiler optimality. A negative autoregressive result despite acceptable teacher-forced PPL is a substantive failure mode, not a metric nuisance.

Protocol SHA256: `a780a26680987fcee2a2e6bc2d1f8247c65cea71fb0f8b9696c5c2566647c504`