> 歴史的な事前計画の科学部分を抜粋した資料。本文の準備・未実行という時制は当時のもの。現在の実行状態は結果報告を参照。

> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Utility-gap exploratory source preparation v1

**SOURCE PREPARATION: GO. SCIENTIFIC EXECUTION: NOT GO.**

Prepared on a separate research branch based on `bb30713cca06519a000251a26668309b1a4ef82b`. Original source/results, including all negative utility outcomes, are unchanged. This package is bounded producer/orchestrator source, gate/schema preparation and one-thread synthetic tests: **no real CIFAR/model-artifact loading, fitting, validation/test decoding, network write or CI occurred**. Randomly initialized synthetic ResNet20 instances are used only in tiny tests. A deny-by-default three-phase CLI and TRAIN-ingestion-only producer are now prepared for review, but neither has been run on real data. There is no automatic retry, resume/warm-start interface, official-test extractor, or original-validation evaluation/training path.

The copied proposal context（元資料のみ・この抜粋には未同梱） preserves the original evidence and full interpretation policy. This protocol fixes its full **two widths × two objectives × two within-trajectory budgets**, not an adaptive budget-only fallback. All metrics/thresholds are exploratory descriptive flags, **not formal noninferiority** or independent confirmation.

## Fixed design and isolation

- Planned original teachers: 26102601, 26102602, 26102603, final epoch120, existing >=85% admission. All three original outcomes, including failures, must be terminal and immutable before any follow-on launch. No teacher substitution or retraining.
- Freeze exact original 5,000 calibration IDs against their original little-endian int64 membership hash. Sort by SHA256 of UTF-8 `canaria-utility-gap-v1:<decimal ID>`, integer ID tie-break. First 1,000 diagnostic; remaining 4,000 fit. Publish exact ordered IDs, raw ID hashes and canonical partition-manifest hash before fitting. No label stratification or split search. **No actual ID partition was loaded or frozen during source preparation.**
- Only new Sep-joint students: two residual maps, ranks [8,8] (5,120 weights) or [16,16] (10,240). Same depth, activations, residual additions, 5×5 receptive field; no biases/BN/repair/trainable suffix. Every width starts from default Conv2d initialization under isolated teacher-seed Torch RNG. Each objective gets an independent exact copy. Widths are not initialization-equivalent.
- Endpoint MSE is elementwise mean. Pure KL uses stable log-softmax, teacher probability times teacher-minus-student log probability, summed over classes and averaged over examples, T=1. Targets detached. `DifferentiableFrozenTeacher.suffix_logits` keeps input gradients while all teacher parameters and BN remain EVAL/frozen. Do not use old no-grad `installed_logits` for KL training.
- Batch128, final32 retained: 32 updates/pass, 64 complete passes, exactly2,048 updates per trajectory. Identical per-seed immutable permutation/batch schedule across four cells. Checkpoints1,024/2,048 are within the same fresh trajectory. Adam lr1e-3, betas(.9,.999), eps1e-8, no decay/clipping/schedule; new state. **12 trajectories,24,576 updates,3,072,000 fit-image presentations**; no extra fitting for second budget.
- Log sample-weighted objectives, gradient norm and relative parameter-update norm without adaptation. Save model, initialization, optimizer, RNG states, complete training log and teacher before/after hashes in checksummed checkpoints. No best selection, early stopping, seed replacement, branch pruning or rerun. Nonfinite failures preserve the prior atomic last-finite artifact; lower checkpoints never substitute for missing final checkpoints.
- At each fixed budget compute full-fit centered endpoint NMSE (existing per-feature across-sample variance, eps1e-12) and logit KL. All 24 checkpoint/metric statuses and all12 trajectory statuses must be terminal and immutable before diagnostic images can open. Missing/failed/not-admitted statuses remain in the full census.
- Private diagnostic phase evaluates both fixed checkpoints for every available cell. Freeze all planned per-image prediction/status records before labels open. Retain teacher logits/predictions, student logits/predictions, exact labels and paired disagreement counts. A consumed marker precedes diagnostic images; additional exclusive markers prevent repeated image or label openings, including after failure/restart. No adaptive choice after seeing diagnostics.
- This diagnostic split is only fresh relative to these students. Original pilot students saw the full calibration set, and the design follows pilot results: **not a pristine study-level holdout**, not independent generalization/selection evidence. No follow-on official-test evaluation is requested. The original heldout gate remains separate and unchanged.

## Prelaunch limit clarification (not an extension of any launched job)

No actual run deadline has been extended: no follow-on job exists. Runtime feasibility, especially KL through the frozen suffix, remains unmeasured.





## Historical strict-ingestion blocker (superseded explicitly below)

**Target:** trusted calibration-only artifact producer/import path.


Names/manifests were inspected only. Preflight contains calibration indices and the original archive, but no separately hashed calibration-image cache was found. Independent audits contain validation predictions/labels/endpoint statistics, not a reusable calibration-image artifact. No numerical cache or image payload was opened during this inspection.


Expected eventual payloads: `fit_images.npy` uint8[4000,3,32,32], `diagnostic_images.npy` uint8[1000,3,32,32], and separately sealed `diagnostic_labels.npy` int64[1000], each full-file SHA256-bound in exact manifest order, with original teachertrain normalization unchanged. The fitting capability has no labels/diagnostic accessor. Diagnostic image and label payloads are not read during fitting.

## Prespecified interpretation

Primary mechanistic contrast: width8/MSE1,024→2,048. Width/loss/interactions secondary; report every seed/cell irrespective of primary result.

- Budget sensitivity: both fit and diagnostic NMSE >=10% lower; task relevance additionally >=10/1,000 accuracy gain. Falling fit error alone does not establish utility or exclude overfitting; flatness is not convergence.
- Width sensitivity: at2,048, width16 vs8 under MSE (KL also reported) improves both NMSEs >=10%; task relevance additionally >=10 correct. If width8 budget fit-NMSE absolute relative change <5%, describe “width benefit after apparent plateau,” never proven insufficient expressivity.
- Alignment evidence: at fixed width/budget2,048, KL vsMSE gains >=10 correct and lowers diagnostic logit KL >=10%, with endpoint NMSE equal/worse. If NMSE also improves, alternative objective helps but alignment is not isolated. Loss geometry/gradient-scale confounding remains.
- Zero baseline makes relative reduction undefined, not a success. No significance/power/NI claim. n=3 teacher seeds share images; missing cells cannot become an available-case n=3 cohort. Mixed/subthreshold/missing outcomes remain mixed/inconclusive. No result authorizes tuning; no factor helping means unresolved under tested changes, not proof of capacity inadequacy.

## Files and verification scope

- `core.py`: fresh fixed models, shared schedules, differentiable frozen suffix, exact loss/stat primitives.
- `data.py`: hashed calibration-only array capabilities and pinned teacher-metadata admission; no mixed-batch decoder.
- `runtime.py`: finite step/log/optimizer/RNG/checkpoint primitives and explicit terminal accounting; no launch/resume CLI.
- `diagnostic.py`, `report.py`: fixed checkpoint verification, label-free prediction/freeze, exact paired counts and exploratory contrasts.
- `test_synthetic.py`: one-thread fake-payload/tiny-network tests. Checkpoint boundary tests construct explicitly synthetic, internally consistent log/Adam/RNG fixtures; they do **not** claim1,024/2,048 actual optimization steps or measured real-data feasibility.



## Independent-review guard repair (source only)

The first prepared source commit `a58d6ea` had six real guard defects. Their original/reproduced probe receipts and the initially failing regression tests remain under `evidence/review_findings/`; passing earlier tests did not make those guards correct. No scientific run occurred under that code.

The repaired preparation now:

- rejects post-terminal metrics/checkpoint operations and returns a stable copied terminal record;
- latches full-fit measurement exceptions as failed checkpoint/trajectory outcomes, distinguishing timeout/cap exhaustion from non-timeout failures;
- requires complete finite NMSE/KL sufficient statistics, consistent centered-NMSE arithmetic, exact session/cell/update/model/fit-ID/teacher bindings, and this session's internally generated one-use metric receipt before any checkpoint write;
- validates full per-update log length/order/sample weighting, default-initialization hash, parameter shapes/dtypes, exact fresh-Adam recipe/state/step census, RNG snapshots and teacher-state identity before serializing or admitting checkpoints;
- requires and hashes the actual canonical full-fit metric bytes again when admitting a diagnostic checkpoint, not just a claimed metric hash;
- rejects NaN/infinite/boolean times and noninteger class predictions/labels, and refuses reservations shorter than the full next two-hour allocation.



A subsequent narrow review found that generic reader `TimeoutError` was still labelled cap exhaustion despite7,199 seconds remaining. Its red receipt is preserved. Runtime now classifies only `protocol.DeadlineExceeded`, emitted by owned immutable clock guards, as cap-exhausted. Ordinary reader/target/OS `TimeoutError` remains failed. The hard-deadline adapter below emits that dedicated signal only for its actual owned deadline; no deadline or scientific policy is changed.


## Explicit PRELAUNCH ingestion amendment and bounded orchestrator

`producer.py` checks the original data-manifest full-file SHA256, full archive SHA256/MD5 on the same descriptor, all five TRAIN-member hashes, original calibration membership and frozen teachertrain normalization. Its restricted pickle reader permits only the official NumPy reconstruction types. It emits separate raw fit4,000/diagnostic1,000 image files and sealed diagnostic-only labels, private0600 files under0700 output directories. `artifact_manifest.json` is committed last, with archive/member/split/provenance hashes and explicit ingestion-only/no-noncalibration-use/no-test-open flags. Partial outputs cannot be resumed or admitted by a successful-manifest inference alone.

The new `entry.py`, `launch.py`, and `worker.py` are a bounded three-phase runner:




