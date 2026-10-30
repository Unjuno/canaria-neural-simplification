# Utility-gap exploratory source preparation v1

**SOURCE PREPARATION: GO. SCIENTIFIC EXECUTION: NOT GO.**

Prepared on a separate research branch based on `bb30713cca06519a000251a26668309b1a4ef82b`. Original source/results, including all negative utility outcomes, are unchanged. This package is in-memory source, gate/schema preparation and one-thread synthetic tests: **no real CIFAR/model-artifact loading, fitting, validation/test decoding, network write or CI occurred**. Randomly initialized synthetic ResNet20 instances are used only in tiny tests. There is no execution CLI, real-data producer, automatic retry, resume/warm-start interface, official-test reader or original-validation reader here.

The copied [proposal context](PROPOSAL_CONTEXT.md) preserves the original evidence and full interpretation policy. This protocol fixes its full **two widths × two objectives × two within-trajectory budgets**, not an adaptive budget-only fallback. All metrics/thresholds are exploratory descriptive flags, **not formal noninferiority** or independent confirmation.

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

The initial proposal's “2 h per teacher including later all-cohort diagnostics” conflicted with delayed diagnostic release. Before launch or freeze, the parent explicitly revised the prospective allocation:

1. A **single immutable cohort clock**, created before numerical imports/artifact loading: <=6 h total, no later than **2026-10-29 06:50 UTC**.
2. Per-teacher **fitting plus full-fit metrics**: absolute <=2 h from that teacher's launch, including loading and downtime; no reset/extension. All fitting ends at least30 min before the cohort/global expiry.
3. Separate all-cohort diagnostic reporting phase: <=30 min, inside the same six-hour/global cap. Waiting for other teachers does not reset a fitting clock and does not extend the overall clock.
4. Conservatively require the full two-hour teacher allocation to remain before starting a new teacher; otherwise preserve explicit missing cells. No shortened window is represented as a complete planned trajectory. Thus6 h is a maximum allocation, not a guarantee of completing three teachers plus diagnostics.
5. One four-thread worker, inter-op1, global science threads<=8; original work has priority. A current parent-reserved resource-slot manifest covering at least the full next two-hour teacher allocation is required; a future orchestrator must acquire the existing shared replacement-worker lock, validate live resources, install the hard deadline **before importing numerics**, and hold the lock throughout each phase. No production orchestrator was added or launched in this preparation.

No actual run deadline has been extended: no follow-on job exists. Runtime feasibility, especially KL through the frozen suffix, remains unmeasured.

## Manifest and authority boundary

`protocol.execution_gate` defaults DENY. An out-of-band reviewed approval hash must bind the exact design, source hashes, original locked cohort/evaluator identity, original calibration membership, new partition, audited calibration-only payload manifest, announced parent approval and current resource reservation. Hashes verify bindings, not authorization by themselves; callers must not manufacture expected approval hashes. The included template is explicitly NOT GO and cannot pass this gate.

`protocol.release_gate` requires a separate out-of-band private diagnostic release approval bound to the immutable full 24-checkpoint cohort. It permits neither original validation nor official test. `data.open_diagnostic_vault` requires the durable consumed marker and creates an exclusive image-opened marker before payload access. Label release additionally checks the persisted exact prediction freeze, complete three-teacher/24-checkpoint census, matching cohort hash and active diagnostic clock, and creates an exclusive label-opened marker. Crashes produce incomplete one-look results, not automatic retry permission.

Python in-memory objects are not a security sandbox against deliberately malicious callers. Trusted orchestration, approval sourcing, filesystem custody, live source verification and artifact provenance still require independent review. Production admission preserves the existing pinned teacher identity/quality checks without the old data loader.

## Existing-data boundary: exact unresolved blocker

**Target:** trusted calibration-only artifact producer/import path.

**Reason:** the old `load_approved_data`/`decode_train_members` materializes mixed CIFAR training batches containing original validation rows. The new follow-on's stricter boundary prohibits decoding those rows, even without evaluating them. That old loader was allowed by the prior protocol; this is **not** evidence that prior results were invalid or leaked.

Names/manifests were inspected only. Preflight contains calibration indices and the original archive, but no separately hashed calibration-image cache was found. Independent audits contain validation predictions/labels/endpoint statistics, not a reusable calibration-image artifact. No numerical cache or image payload was opened during this inspection.

**Pending:** either find an already authorized, integrity-bound calibration-only image cache, or explicitly review a narrow *ingestion-only selection* amendment that permits transient mixed-batch decoding solely to select approved calibration rows, never evaluating/training/using original validation rows. That amendment has **not** been applied here. Under the current strict protocol, the producer remains blocked. No silent weakening, execution attempt or workaround occurred.

Expected eventual payloads: `fit_images.npy` uint8[4000,3,32,32], `diagnostic_images.npy` uint8[1000,3,32,32], and separately sealed `diagnostic_labels.npy` int64[1000], each full-file SHA256-bound in exact manifest order, with original teachertrain normalization unchanged. The fitting capability has no labels/diagnostic accessor. Diagnostic image and label payloads are not read during fitting.

## Prespecified interpretation

Primary mechanistic contrast: width8/MSE1,024→2,048. Width/loss/interactions secondary; report every seed/cell irrespective of primary result.

- Budget sensitivity: both fit and diagnostic NMSE >=10% lower; task relevance additionally >=10/1,000 accuracy gain. Falling fit error alone does not establish utility or exclude overfitting; flatness is not convergence.
- Width sensitivity: at2,048, width16 vs8 under MSE (KL also reported) improves both NMSEs >=10%; task relevance additionally >=10 correct. If width8 budget fit-NMSE absolute relative change <5%, describe “width benefit after apparent plateau,” never proven insufficient expressivity.
- Alignment evidence: at fixed width/budget2,048, KL vsMSE gains >=10 correct and lowers diagnostic logit KL >=10%, with endpoint NMSE equal/worse. If NMSE also improves, alternative objective helps but alignment is not isolated. Loss geometry/gradient-scale confounding remains.
- Zero baseline makes relative reduction undefined, not a success. No significance/power/NI claim. n=3 teacher seeds share images; missing cells cannot become an available-case n=3 cohort. Mixed/subthreshold/missing outcomes remain mixed/inconclusive. No result authorizes tuning; no factor helping means unresolved under tested changes, not proof of capacity inadequacy.

## Files and verification scope

- `protocol.py`: stdlib manifest, partition, cohort/teacher/release caps and frozen-status gates.
- `core.py`: fresh fixed models, shared schedules, differentiable frozen suffix, exact loss/stat primitives.
- `data.py`: hashed calibration-only array capabilities and pinned teacher-metadata admission; no mixed-batch decoder.
- `runtime.py`: finite step/log/optimizer/RNG/checkpoint primitives and explicit terminal accounting; no launch/resume CLI.
- `diagnostic.py`, `report.py`: fixed checkpoint verification, label-free prediction/freeze, exact paired counts and exploratory contrasts.
- `test_synthetic.py`: one-thread fake-payload/tiny-network tests. Checkpoint boundary tests construct explicitly synthetic, internally consistent log/Adam/RNG fixtures; they do **not** claim1,024/2,048 actual optimization steps or measured real-data feasibility.

Remaining execution prerequisites: original completed lock; exact published real-ID partition; audited data boundary/producer; production orchestration and all source/dependency bindings; independent GO; announced execution approval; live reserved CPU slot; all-cohort private release approval. Source preparation does not satisfy these automatically.


## Independent-review guard repair (source only)

The first prepared source commit `a58d6ea` had six real guard defects. Their original/reproduced probe receipts and the initially failing regression tests remain under `evidence/review_findings/`; passing earlier tests did not make those guards correct. No scientific run occurred under that code.

The repaired preparation now:

- rejects post-terminal metrics/checkpoint operations and returns a stable copied terminal record;
- latches full-fit measurement exceptions as failed checkpoint/trajectory outcomes, distinguishing timeout/cap exhaustion from non-timeout failures;
- requires complete finite NMSE/KL sufficient statistics, consistent centered-NMSE arithmetic, exact session/cell/update/model/fit-ID/teacher bindings, and this session's internally generated one-use metric receipt before any checkpoint write;
- validates full per-update log length/order/sample weighting, default-initialization hash, parameter shapes/dtypes, exact fresh-Adam recipe/state/step census, RNG snapshots and teacher-state identity before serializing or admitting checkpoints;
- requires and hashes the actual canonical full-fit metric bytes again when admitting a diagnostic checkpoint, not just a claimed metric hash;
- rejects NaN/infinite/boolean times and noninteger class predictions/labels, and refuses reservations shorter than the full next two-hour allocation.

Consistency checks cannot prove that a malicious producer really performed optimization; independently trusted producer/source/artifact custody remains essential. The two-hour reservation gate is admission validation, not a substitute for production lifetime resource monitoring, exclusive locking or the hard-deadline orchestrator, all still pending. Synthetic audit fixtures are explicitly constructed consistency fixtures, never evidence of1,024/2,048 measured numerical updates. No scientific design, threshold, data boundary, deadline or launch authority changed in this repair.


A subsequent narrow review found that generic reader `TimeoutError` was still labelled cap exhaustion despite7,199 seconds remaining. Its red receipt is preserved. Runtime now classifies only `protocol.DeadlineExceeded`, emitted by owned immutable clock guards, as cap-exhausted. Ordinary reader/target/OS `TimeoutError` remains failed. The pending hard-deadline adapter must emit that dedicated signal only for its actual owned deadline; no deadline or scientific policy is changed.
