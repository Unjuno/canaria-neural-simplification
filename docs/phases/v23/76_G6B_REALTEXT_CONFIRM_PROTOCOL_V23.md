# G6b real-text causal LM confirmatory protocol — v23

## Locked condition
Selected from the prespecified pilot menu by the written selection rule:
- teacher: 4 causal Transformer blocks, d=24, 4 heads, MLP=48;
- compiled core: 2 causal Transformer blocks, d=24, 4 heads, MLP=24;
- 512 unlabeled train windows for residual-stream MSE fit;
- tau=8 **joint repair** of compiler + embedding/position/final norm/lm head at lr=3e-4;
- matched uncompiled control receives the same 8 continuation epochs;
- whole-model parameter reduction must remain >=35%.

Pilot seeds 3497, 3498, 3499 are excluded from confirmatory inference.

## Data
Natural English text comes from scikit-learn's locally distributed dataset-description `.rst` documents. Document identities are fixed before window sampling:
- train: breast_cancer, california_housing, covtype, diabetes, digits, iris, kddcup99, lfw, linnerud;
- validation: olivetti_faces, rcv1;
- test: species_distributions, twenty_newsgroups, wine_data.

Character normalization/tokenization and window seeds are fixed in the runner. No document crosses splits.

## Cohort
Use the first 8 seeds >=3500 satisfying, before compilation:
- validation PPL <=20.0;
- validation token accuracy >=0.20.

Ineligible seeds are excluded solely by this baseline rule. Continue sequentially until 8 eligible seeds are obtained.

## Primary transfer decisions
Parameter reduction must be >=35%.

**Zero-shot Z PASS** only if both seed-bootstrap 95% lower bounds satisfy:
- tau0 PPL utility = control/base PPL divided by compiled PPL >=0.95;
- tau0 greedy rollout token agreement to the uncompiled baseline >=0.90.

**Adapted A PASS** only if both seed-bootstrap 95% lower bounds satisfy:
- tau8 matched-control PPL utility >=0.95;
- tau8 greedy rollout token agreement to the matched continued-training control >=0.90.

50,000 seed bootstrap resamples; seeds are the inference unit.

Greedy rollout agreement remains required even though pilot results were poor. The threshold is not weakened after pilot observation.

## Secondary
Report exact rollout agreement and mean first divergence position. If functional adapted transfer passes, run q8+zlib state-stream accounting as a separate follow-up. If functional adapted transfer fails, storage follow-up is optional and cannot rescue the transfer decision.
