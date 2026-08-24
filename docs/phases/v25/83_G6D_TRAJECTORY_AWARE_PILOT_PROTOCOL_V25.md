# G6d Trajectory-aware compiler pilot protocol — v25

## Question
Does bounded training on compiler-generated prefixes prevent the natural-text autoregressive trajectory divergence that survived hidden-state MSE (v23) and teacher-forced logit KL (v24)?

## Frozen model/capacity
Same v23/v24 natural-English character LM, document split, 4-block teacher, and 2-block/MLP24 compiler. Compiler parameter count is unchanged (11,042 total compiled-model parameters vs 23,138 teacher; 52.2776% reduction).

No ground-truth next-token labels are used in compiler fitting or trajectory refinement. The teacher supplies hidden states/logits only.

## Discovery seeds
3697 and 3698 only; excluded from confirmatory inference.

## Equal-update candidate menu
All candidates use 320 total compiler optimizer updates.

### T0 — teacher-prefix O1 control
20 O1 epochs on 512 unlabeled data windows (16 minibatches/epoch = 320 updates): normalized hidden MSE + teacher-forced logit KL.

### T1 — 8-step one-iteration trajectory distillation
- 15 O1 warm-start epochs = 240 updates.
- Generate greedy prefixes from the warm-start compiler using 64 fixed training prompts of length 12 for horizons 1..8.
- Perform exactly 80 refinement updates: 40 teacher/data-prefix O1 updates alternating with 40 on-policy prefix updates.
- On-policy update loss is last-position normalized hidden MSE + teacher-logit KL on the compiler-generated prefix.
- Prefix pool is generated once and is not regenerated after refinement begins.

### T2 — 24-step one-iteration trajectory distillation
Identical to T1 except on-policy prefixes cover horizons 1..24.

## Selection
Measure tau=0 test PPL utility and 24-token greedy rollout agreement vs teacher.
Select highest mean rollout agreement across discovery seeds subject to mean PPL utility >=0.98. If within 0.01, choose simpler T0 < T1 < T2.

## Confirmation
Freeze the selected condition before any seed >=3700 is inspected. Use first 8 seeds >=3700 meeting validation PPL <=20 and token accuracy >=0.20.

Primary zero-shot PASS requires lower 95% seed-bootstrap CI >=0.95 for PPL utility AND >=0.90 for 24-token rollout agreement.

If zero-shot fails, no additional repair adaptation is permitted in v25: v23/v24 already established that the bounded tau8 joint repair does not solve this natural-text regime. This phase isolates the compiler objective.

No storage optimization unless zero-shot functional transfer passes.
