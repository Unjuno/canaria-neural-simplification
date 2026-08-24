# G6b real-text causal LM results — v23

## Purpose
Test whether the small decoder-only Transformer simplification observed on the synthetic causal language in v22 transfers to held-out natural English text.

## Data and model
Natural English text is taken from `.rst` dataset-description documents distributed locally with scikit-learn. Documents, not windows, are split across train/validation/test sets before sampling. The model is a character-level causal Transformer.

Teacher:
- 4 causal Transformer blocks;
- d=24, 4 heads, MLP=48;
- 23,138 parameters.

Selected compiled condition after the bounded pilot menu:
- 2 causal Transformer blocks;
- d=24, 4 heads, MLP=24;
- 11,042 parameters;
- **52.2776% whole-model parameter reduction**;
- residual-stream MSE fit on 512 unlabeled train windows;
- tau=8 joint repair of compiler plus embedding/position/final norm/lm head;
- matched uncompiled control gets the same continuation epochs.

Pilot seeds 3497-3499 are excluded. Confirmatory seeds are the first 8 seeds >=3500 satisfying validation PPL <=20 and token accuracy >=0.20: **3500-3507**.

## Confirmatory results
50,000 seed bootstrap resamples; seed is the inference unit.

| condition | metric | mean | 95% CI | threshold | decision |
|---|---|---:|---:|---:|---|
| tau=0 | PPL utility | **0.99703** | **[0.99576, 0.99816]** | lower95 >=0.95 | PASS |
| tau=0 | greedy rollout token agreement | **0.63265** | **[0.55029, 0.72689]** | lower95 >=0.90 | **FAIL** |
| tau=8 joint repair | matched-control PPL utility | **0.94577** | **[0.94115, 0.95004]** | lower95 >=0.95 | **FAIL** |
| tau=8 joint repair | greedy rollout token agreement | **0.41130** | **[0.33887, 0.48551]** | lower95 >=0.90 | **FAIL** |

Secondary rollout measures:
- tau=0 exact continuation agreement: 0.37891 [0.23828, 0.53906]
- tau=8 exact continuation agreement: 0.10938 [0.03125, 0.19922]
- tau=8 mean first-divergence position over the 24-character rollout: 5.996 [3.785, 8.492]

### Decision
- **Zero-shot transfer: FAIL.**
- **Adapted transfer: FAIL.**
- Generalization label: **N — no transfer under the tested adaptation budget.**

No q8/storage follow-up is used to rescue the decision because the functional transfer criterion failed.

## Main scientific observation
The replacement preserves teacher-forced next-character perplexity remarkably well at tau=0 (utility 0.997), while the free-running greedy trajectory diverges strongly (agreement 0.633). This is not a small statistical effect: the rollout-agreement confidence interval remains far below the preregistered 0.90 threshold.

Moreover, the selected joint-repair adaptation does not repair the mismatch. Relative to a continued-training control, tau=8 PPL utility falls to 0.946 and rollout agreement to 0.411.

Therefore **local next-token likelihood preservation is not sufficient evidence that an autoregressive computation has been successfully simplified**. Small logit changes can preserve teacher-forced loss while changing the state/input trajectory induced by the model's own generated tokens.

This strengthens the evaluation lesson first seen in v22: autoregressive systems require rollout-sensitive metrics. In v23 the issue survives a shift from deterministic synthetic language to held-out real English prose.

## Interpretation relative to v20-v22
- v20 small ViT: A — adapted transfer.
- v21 non-image Transformer encoder: Z — zero-shot transfer.
- v22 synthetic causal decoder LM: A — adapted transfer.
- **v23 real-text character LM: N — no transfer under tested budget.**

Thus the emerging map is explicitly mixed. Transformer simplification is not uniformly positive; autoregressive natural-text dynamics expose a failure regime that is largely invisible to teacher-forced PPL.

## Limitations
- The corpus is small, domain-specific scikit-learn documentation, not a large natural-language benchmark.
- The model is a small character-level LM trained from scratch, not a pretrained subword LM.
- The negative result applies to the preregistered 4->2-block candidate family and bounded adaptation menu; it does not prove that no alternative compiler can work.
- Greedy rollout agreement is a strict functional-fidelity metric and can amplify small logit differences. That sensitivity is intentional for this phase because free-running dynamics are part of the system being compressed.

## Next discriminative experiments
1. Measure rollout divergence as a function of horizon (1,2,4,8,16,24 tokens) to locate the error-amplification timescale.
2. Test whether a compiler objective that includes one-step logit/KL matching, rather than residual-stream MSE alone, reduces rollout divergence without increasing model size substantially.
3. Only after a fixed objective is selected on pilots, test a small subword/pretrained real-text LM on fresh checkpoints.
4. Keep teacher-forced and rollout criteria jointly primary; do not accept a PPL-only success.
