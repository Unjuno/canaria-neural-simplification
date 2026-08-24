# G6 Small Decoder-Only Causal LM Pilot Protocol — v22

## Question
Does the Transformer simplification recipe selected before G5 transfer to a causal decoder-only next-token model, including autoregressive generation, without architecture-family-specific compiler search?

## Evidence role
Pilot/smoke only. Seeds below 3400 are never used in confirmatory inference.

## Controlled language
A deterministic prompted synthetic language is used to isolate causal/autoregressive computation without external-data or tokenizer confounds.

Each length-20 sequence contains a 4-token prompt: `[BOS, MODE, KEY, START]`. The remaining 16 tokens are generated deterministically from the previous data token, MODE, KEY, and position parity. Training loss is computed only on continuation tokens.

Separate fixed train/validation/test generators are used. Validation is used only for baseline eligibility. Confirmatory metrics are reported on a distinct test split.

## Teacher architecture
- decoder-only causal Transformer
- depth 4
- d_model 24
- 4 attention heads
- MLP width 48
- learned token and position embeddings
- LayerNorm + untied LM head

## Frozen transferred compiler recipe
- replace all 4 teacher blocks with 2 causal Transformer blocks
- same d_model and head count
- compiler MLP width 24
- fit compiler by residual-stream activation MSE on 512 unlabeled training sequences
- compiler fit never uses token labels/loss
- compiler frozen during any shell repair

## Metrics
Primary LM utility is `control_perplexity / compiled_perplexity`.
Secondary metrics include teacher-forced continuation token accuracy, greedy autoregressive continuation token accuracy, exact continuation sequence-match rate, first-error position, and parameter reduction.

## Repair ladder
- tau=0: no task repair
- tau=2 and tau=8: only token embeddings, position embeddings, final norm and LM head may train; compiled core remains frozen
- matched continued-training controls receive the same task epochs

## Pilot seed
3399 only.

## Confirmatory trigger
If pilot establishes an eligible baseline and numerically stable compile path, write and hash-lock a separate confirmatory protocol before inspecting any seed >=3400 outcome.