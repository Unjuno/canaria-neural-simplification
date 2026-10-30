> 歴史的な事前計画の科学部分を抜粋した資料。本文の準備・未実行という時制は当時のもの。現在の実行状態は結果報告を参照。

> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Conditional seen-digits strong-control budget extension, fixed v1


## Question and evidence class
Does single-map Comp require a smaller first-passing candidate-grid budget than jointly fitted two-map J, on the sixteen already observed teachers? This is an adaptive follow-on, conditional on reused digits/train-validation data, not independent confirmation, objective-only causality, global minimum capacity, or deployment compression. Original Comp/local ratio remains a different comparator.

## Fixed design
Reuse all sixteen teachers, fit targets and 600×128 sample schedules from the completed matched study runtime 6d9e60a. PARENT.json binds every original artifact byte and the inherited helper. No teacher training or 2048 student rerun. Original 2048 C/J metrics are copied verbatim with provenance, not new replicates.
Budgets 512,1024,1536,2048,3072,4096,6144 are fixed, two J maps width h and one C map width 2h, h=2,4,6,8,12,16,24. Only six missing widths are fitted: 16×6×2=192 new fits. Width/arm initialization seeds are fixed in SEEDS.json and collision-checked. Both arms consume the parent's exact schedule and optimizer: AdamW lr .008, weight_decay .00001, betas .9/.999, eps 1e-8, all other flags explicit; 600 updates, 76800 sampled examples per parameter, scoped float64 sqrt profile. No early success stopping, adaptive widths, checkpoint selection or hyperparameter changes. Equal parameters/updates are not equal compute, and C/J initialization shapes cannot match.
Fit all twelve new replacements for a teacher before validation. Reconstruct only original train/validation partitions (full raw digits used only for original stratification), verify original tensor hashes. Load owned NPZ with allow_pickle=False, strict state keys/shapes/dtypes and finite values. Verify initial and final teacher state hashes, frozen parameters, input/target/schedule immutability, and exact parent validation target/logits/labels. No test subset, inference or metrics. New evaluation uses fixed final weights only. Store state, update losses, endpoints/logits and scalar metrics for each cell.

## Locked outcome and censoring
Each cell passes iff span NMSE≤.08 and validation accuracy≥same teacher accuracy−.02. Evaluate every cell; no monotonic interpolation. First passing grid budget is the smallest observed passing candidate, not a continuous capacity minimum. All sixteen teachers retained, regardless of teacher accuracy or student success.
All 16×7×2=224 cells (32 reused +192 new) required with exact identities. Missing/numerical/process/integrity failure => INCOMPLETE; never exclude, replace or retry. For complete grids report every cell, per-budget pass counts, nonmonotonic pass patterns, all 16 minima, and both-pass/C-only/J-only/neither coverage. No-pass means no observed success through6144 under fixed recipe, not mathematical impossibility or guaranteed larger-budget behavior.
Primary ratio only if BOTH arms pass some grid point for ALL16 teachers. Compute mean log2(minBudget_C/minBudget_J); bootstrap100000 paired-seed resamples with numpy default_rng(2710276144), percentile95%. Upper<0 => CONDITIONAL_GRID_ADVANTAGE; lower>0 => CONDITIONAL_GRID_DISADVANTAGE; otherwise UNCERTAIN. Geometric ratio and interval are descriptive conditional-on-seen-cohort summaries. If any censored arm, decision CENSORED_NO_COHORT_RATIO; no complete-case rescue or favorable CI. C-only/J-only/neither reported, no made-up infinity ratios. Precision/feature differences at all budgets descriptive, not additional pass tests. Parent2048 outcomes remain identifiable as reused.

