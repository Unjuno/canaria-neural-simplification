> 歴史的な事前計画の科学部分を抜粋した資料。本文の準備・未実行という時制は当時のもの。現在の実行状態は結果報告を参照。

> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Fixed direct-Comp decision-preservation protocol v1

**PREPARATION ONLY — execution NOT GO.** This new isolated protocol does not
modify, retry or extend any previous spatial/utility-gap/official-test experiment.
The exact scientific design is the adjacent byte-preserved `PROPOSAL.md`; this
file describes its implementation and gates, not an alternative experiment.

## Fixed treatment

Three original teachers: 26102601, 26102602, 26102603. One direct rank16 Comp map
per arm, exactly 5,120 bias-free weights, span [4:6]. Reconstruct the original
Sep-first/Comp-second RNG order and deep-copy Comp for feature and decision arms.
Both receive exactly 2,048 fresh-Adam updates, 64 complete 4,000-image passes,
128 samples except the fixed final 32, same immutable seed-specific order. No
augmentation, labels, warm-start, schedule change, clipping, retry or selection.

F is ordinary endpoint MSE over batch/channel/space. D is **only** the proposal's
float64, temperature-one, all-nine-rival probability-weighted one-sided squared
teacher-gap erosion divided by batch count and the **single full-fit mean rival
mass μ**. No KL, feature blend, cutoff, searched coefficient or temperature.

Teacher logits, q, gaps and μ are detached. D student logits use the frozen
suffix **with input autograd enabled**, never `installed_logits(no_grad)`.
Teacher parameters, BN/eval state and whole state hashes are invariant. Every
positive rival weight must remain finite and positive; underflow/nonfinite loss
D loss/gradient is a legitimate saturated objective and receives the prescribed
Adam update; no early stop follows. Missing gradients still fail.

### Zero-loss statement and its limits

With strictly positive rival weights, zero mathematical D loss implies every
student top-relative gap is at least its teacher gap. Later tied teacher rivals
may stay tied; every rival preceding the teacher's first-index winner has a
strictly positive teacher gap, so the same deterministic first-index class is
preserved. This does **not** preserve the soft distribution: expanded gaps have
zero loss, may inflate logits, and can preserve teacher mistakes. Numerical
underflow is rejected. Small nonzero loss is not a guarantee of any decision,
and fitting says nothing automatically about new examples. Objective geometry,
uncertainty emphasis, scale/conditioning and compute remain joint confounds.

## Data/target order and custody

Only the existing hash-bound 4,000-fit/1,000-diagnostic calibration cache is
reachable. No archive ingestion or original-validation/official-test reader is
provided. Those historical outcomes remain closed and negative records intact.

2. A bounded `strata` child opens only the old gap run's already-saved **label-free
   predictions** plus its frozen cohort. It verifies exact diagnostic IDs and
   teacher checkpoint/state identities, extracts only teacher logits (old student
   fields are discarded), and freezes all three teachers' five 200-image rank bins
   sorted by `(teacher top1−top2 margin, image ID)`. Exact IDs/hashes and ties at
   all four edges persist **before the first fit**. This prefit phase consumes
   teacher01's original two-hour clock, not a free extra allocation.
3. Per teacher, one ordered 4,000-fit pass creates detached teacher targets and
   μ. The full target payload and identities are exclusively frozen before any
   optimizer is constructed. The two trajectories share this immutable target
   bank; each checkpoint includes and binds it. Reconstructing q/μ during
   validation is only an identity check, not changing the stored optimizer loss.
4. Fit steps receive fit-only images/IDs, never diagnostic logits/strata values,
   labels or report metrics. Only an audit hash of the prefrozen strata is in the
   session lock. Diagnostics do not affect loss, sampling, stopping or selection.
5. Freeze all six fixed final-checkpoint/failure records and establish child
   termination before one separately reviewed final report. Failed or capped
   cells remain missing/inconclusive, never substituted or restarted.
6. Final reporting validates all eligible checkpoints/target identities, then
   measures fit and diagnostic endpoints. Teacher logits are checked exactly
   against the prefrozen fit/diagnostic cache under the pinned runtime. All six
   prediction/failure records persist before the diagnostic labels open once.
7. Report all 1,000 examples and all five strata for every surviving arm,
   including gains, harms, lost corrections, shared/unique harms, Q5, ties,
   nine-rival erosion, endpoint NMSE, logit scale and both fit objectives. Partial
   pairs retain surviving-arm details while the comparison stays incomplete.

No labels are used to define bins. These diagnostics were already seen; none of
this restores independence. Descriptive findings cannot become a utility pass,
original-validation pass, official-test result or independent confirmation.

## Fixed flags and resource bound

Per seed: F Q1 harms >0; `5*H_D,Q1 <= 4*H_F,Q1`; Q2–Q5 harms do not increase;
Q5 harms separately do not increase. The task flag additionally requires at
least 10 extra correct/1,000 overall. The cohort prediction is the task flag in
**all three** teachers. Every component is reported; mixed remains mixed;
all-complete failure falsifies this specified prediction. Thresholds are fixed
descriptive targets, not significance/noninferiority/power claims.

Six trajectories, 12,288 updates and 1,536,000 fit-image presentations (target and
report inference additional). One four-thread worker, inter-op1, science threads
at most8 globally. Each teacher has one immutable 2h cap including target prep,
loading/downtime (teacher01 includes initial cached-strata extraction); final
report at most30min; overall at most6.5h /26 CPU-thread-hours and absolute cutoff
2026-10-29 06:50 UTC. The complete allocation must fit before launch. No cap
blocks later workers/reporting rather than pretending termination.

Existing shared replacement-worker lease is reused only for mutual exclusion;
