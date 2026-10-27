> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# R87R2 equal-budget validation control census

## Finding

At the fixed 2,048 learned-weight budget, all 16 original-primary-host R87R2 seeds have validation feature NMSE ordered **single-map Comp < joint-factorized two-map < local componentwise two-map**. This is consistent with the old descriptive control pattern under the R87R2 numerical recipe, not a new confirmatory pass or causal identification.

| Arm | Full definition | Mean validation NMSE | Mean validation accuracy | Original fixed-budget quality count |
|---|---|---:|---:|---:|
| L | Two TinyRes(64,8) maps, each fitted locally to teacher intermediates | 0.1515253841 | 94.74537075% | 0/16 |
| J | Same two-map shape, jointly fitted against span endpoint | 0.0664010310 | 96.22685276% | 12/16 |
| C | One TinyRes(64,16) map fitted against span endpoint | 0.0539574242 | 96.50462978% | 14/16 |

Quality means exactly **validation NMSE ≤ 0.08 AND validation accuracy ≥ the same seed's teacher validation accuracy − 0.02**. Every denominator is all 16 seeds, 871200–871215; none is removed. These are fixed-budget arms, NOT each method's selected smallest passing endpoint. Both J and C satisfy the feature threshold in all 16 seeds, but task accuracy prevents J from satisfying the combined criterion in four seeds and C in two. All L NMSE values exceed 0.08.

J has higher validation accuracy than L in 13 seeds, ties in two, and is worse in one. C versus J is better in eight, tied in five, and worse in three. Thus uniform feature-error improvement does not imply uniform task-accuracy improvement. C versus L is better in 12 and tied in four.

The mean NMSE gap decomposes arithmetically as:

0.0975679599 (L−C) = 0.0851243532 (L−J) + 0.0124436068 (J−C).

After the fixed census, the parent requested relative summaries: the ratio of arm-mean NMSE implies a 56.1783% reduction J versus L and an 18.7401% reduction C versus J. These are descriptive ratios of means, not means of per-seed ratios and not a causal “fraction explained.” They were not the fixed directional criterion.

## Limits and next-design consequence

Equal learned parameter count is exact: L and J contain 2×(64×8+8×64)=2,048, C contains 64×16+16×64=2,048. All parameters receive 600 updates, but L uses two separate optimizers/fit calls, and measured forward/backward compute is not equated. Different initialization seeds, minibatch sequences, and teacher-forced versus student-produced intermediates remain confounded. J−L cannot identify objective alone or covariate shift alone; C−J cannot identify topology alone.

A future untouched-confirmation design must include the joint-factorized control rather than contrasting only local componentwise fitting against Comp. It must explicitly control or randomize initialization/data streams and define compute accounting; this report does not authorize a new fit or establish an untouched confirmation dataset. It neither reopens CIFAR fitting nor repairs Issue87, demonstrates practical whole-model compression, or adds independent seeds to the prior experiment.

## Execution and evidence

- Immutable source commit: 056d69cc257a7670fb45e17f11588afa497d8881.
- Protocol commit: 06eee28 (full identity in VERIFICATION.json); implementation frozen next at 94829b3 before projection/census.
- census.py uses Python standard library only; two separate stages ensure computation consumes validation_input.json only.
- The original scalar container includes test fields. Its bytes were hashed and parsed for the fixed validation allowlist; no test metric was printed, calculated, compared, selected or used for this analysis. No model, dataset, logits, checkpoint or endpoint tensors were opened.
- One projection and one census execution; resource limits 256 MiB address space / 600 CPU seconds and 600-second shell timeout. Execution receipts record actual elapsed time/RSS. No training, inference, installations, CI, network writes or publication.
- artifacts/table.csv and result.json retain every seed, all contrasts/statistics, the eight-cell utility cross-tab and all arithmetic decomposition terms.
- Source/projection/code/output hashes are retained in VERIFICATION.json, with the independent validation-only arithmetic audit recorded separately.
