# PROVISIONAL: three-arm CIFAR-10 replacement comparison, no repair

Preparation only, based on reviewed teacher commit `67bd076`. This document and
source do not authorize experiments. No real data/checkpoint access, fitting or
classification/feature evaluation until parent scientific review, announced plan,
and each teacher's final quality gate. Core has no filesystem/data loader; a separate preparation-only pinned loader and
fit plumbing now exist, with no CLI/autostart. Execution remains unauthorized.
No downloads, CI, pushes, charges, or user-PC use.

## Cohort and binding

Preserve all three planned seeds 26102601/26102602/26102603, including failures;
no replacements, seed selection or outcome-adaptive recipe changes. Final teacher
epoch must equal120, NOT >=85 epochs: >=85 refers to validation percent accuracy.
Admission requires completed validation with >=4250/5000 correct, exact fraction,
adequate=true, final checkpoint full-file and payload SHA256, restricted
weights-only load, planned seed, checkpoint final cursor/order, matching export,
checkpoint, metrics and independently approved preflight identity/binding.
Teacher state tensors must be finite and load strictly into pinned ResNet20.
Expected identity must contain approved architecture/config hash, all split hashes,
normalization, and split seed; hashes are integrity/binding, not authentication.
Source/provenance authorization remains caller responsibility. Failed/unfinished
teachers are reported as such, not silently removed from the intended cohort.

## Fixed intervention

Zero-based half-open `blocks[4:6]` = stage2 blocks2–3. Prefix includes stage2
block1. Input, intermediate and output each32x16x16. Entire teacher, prefix,
suffix, BN running statistics and classifier frozen in EVAL for feature extraction
and installed predictions. Only replacement convolution weights are optimized.

Every BN-free map is `M_r(x) = ReLU(x + Conv1x1(r->32)(ReLU(Conv3x3(32->r,pad1)(x))))`.
Both convolutions have no bias, stride1; no other trainable parameters. Parent
review changed initially proposed GELU/no-postactivation to inner and post-add
ReLU before any fit/outcome, to match teacher activation boundaries. This is a
pre-outcome design choice, not a result-dependent modification.

| Arm | Architecture | Fit target | Convolution weights |
|---|---|---|---:|
| Comp-joint | one map, r16 | common span endpoint |5120|
| Sep-local | two maps, each r8 | teacher-forced per-block endpoints |5120|
| Sep-joint | same two maps, each r8 | common span endpoint, own intermediate |5120|

Torch default Conv2d initialization (Kaiming-uniform with a=sqrt(5)); local and
joint separate arms are independent deep copies of one exact initial state.
Persist and hash that shared snapshot before future fitting. Comp initialization
comes from the same seeded isolated RNG after Sep construction; equality across
different architectures is neither possible nor claimed. Use seed-specific
replacement initialization and schedule seed equal to the teacher seed, without
search. Snapshot each arm, recipe and ID schedule before exposure to outcomes.

**Unavoidable architecture confounds:** Comp maximum receptive field3x3, Sep5x5,
teacher replaced span9x9; depth, residual additions and number of nonlinearities
also differ. Equal parameter count does not imply equal compute, capacity,
receptive field, or function class. Sep-local versus Sep-joint isolates training
objective/dataflow given identical architecture/init; Comp comparison does not
isolate composition alone. No causal/general proof from a directional n=3 pilot.

## Provisional fixed fitting recipe — parent approval still required

- Calibration only: all5000 admitted calibration images, no labels in fitting.
  Deterministic original32x32 images, no augmentation; teachertrain-only saved
  normalization, no recomputation using calibration/validation/test.
- Teacher endpoints `(x,t1,t2)` computed with the fixed frozen EVAL teacher.
- Exactly1000 Adam updates per arm, batch128, retain each epoch's final8 samples.
  Every40 updates covers a freshly permuted5000 IDs;1000 updates =25 complete
  calibration passes =125000 image presentations per arm.
- Adam lr1e-3, betas(.9,.999), eps1e-8, weight_decay0, constant schedule; no clipping,
  early stopping, validation-based selection, restarts or outcome tuning. Same
  optimizer implementation/settings and immutable ID batch tuple for all arms.
- Sep-local loss: `mean((M1(x)-t1)^2) + mean((M2(t1)-t2)^2)`.
  Each mean is over batch/channel/space. Sum, not average, so each module receives
  its standalone local-MSE gradient scale. Both modules update every batch through
  one optimizer. Teacher features detached; no predicted-intermediate local fit.
- Sep-joint: `mean((M2(M1(x))-t2)^2)`; gradient propagates through both maps.
  Comp-joint: `mean((M(x)-t2)^2)`. No head/stage3/BN/suffix repair whatsoever.
- Future execution must check finite parameters, gradients, optimizer state and
  outputs; record failures without changing recipe. Partial/crashed fits are not
  admissible final1000-update outcomes. Production checkpoint/resume and hard caps are now prepared but NOT RUN; see
  execution-plumbing section below. They require separate review before science.

## Outcomes and sealed test

Primary no-repair outcomes: installed classification accuracy and **centered**
common-span NMSE on identical held-out images. For teacher endpoint `Y[n,c,h,w]`,
compute mean over the full sample axis N separately at every `(c,h,w)`, then
`SSE / (sum(Y^2) - sum_features(sum_samples(Y)^2/N) + 1e-12*N*C*H*W)`.
This equals mean squared error divided by mean centered teacher variance plus
1e-12. Accumulate float64 sample sums/squared sums and SSE over all batches;
never average per-batch ratios. Tiny roundoff-negative variance is clamped to0;
material negative/nonfinite statistics fail. A constant teacher endpoint has the
explicit epsilon floor. This definition was revised BEFORE outcomes. The old
uncentered `SSE/sum(Y^2)` is retained only as named secondary
`uncentered_energy_nmse` (undefined zero energy => null). No historical NMSE
threshold is reused; both are descriptive.

All arms use actual installed span endpoints, including Sep-local composition.
Never compare local training loss to joint loss as evidence. Primary contrast:
Comp-joint versus Sep-local accuracy; Sep-joint versus Sep-local isolates objective
with architecture/init fixed; Comp-joint versus Sep-joint measures architecture
tradeoff, not composition alone. Report all planned seed outcomes individually.

Pre-outcome exploratory task-utility criterion: final1000-step no-repair accuracy
drop versus the same admitted teacher <=2 percentage points. On5000 validation
images this is exactly `replacement_correct >= teacher_correct - 100`; on a
future separately approved10000-image test, `>= teacher_correct - 200`. This is a
descriptive pilot criterion, NOT a statistical noninferiority proof. Record teacher
accuracy, replacement accuracy and paired percentage-point differences per seed.
No repair outcomes mixed into this comparison; no seed substitutions or n3
population-generalization claims.

Fixed5000 validation images may be used only after admission and plan approval,
for preregistered final-only reporting, not tuning or checkpoint selection.
Official test remains sealed until recipes, teacher/replacement checkpoint hashes,
all failures, and validation outcomes for the intended cohort are locked in a
manifest. A separate parent-approved test-release action is required; no helper
here can read test data. No official-test access during preparation or fitting.
A future repair experiment, if desired, is separate and cannot revise this one.

## Prepared execution plumbing — NOT RUN, not an execution authorization

- `data.py` pins teacher source/spec and preflight SHA metadata from source
  manifest `cfcc082`, whose teacher code is67bd076. Only source files/manifest
  text were inspected in preparation. Official archive MD5 and pinned SHA256,
  individual training member hashes, exact reconstructed/saved partition IDs and
  teachertrain-only normalization must match. Archive is hashed then rewound on
  the same descriptor; only five training members are decoded, never test_batch.
- Establish preliminary final teacher quality gate from export/metrics before
  data access; after independently reconstructing approved data, production
  admission additionally verifies exact teachertrain membership and normalization
  tensors, full recipe, source/runtime and strict teacher state. A boolean API
  guard is not provenance authentication or authorization by itself.
- Caller first invokes standard-library `launch.open_launch` and `arm_deadline`
  BEFORE numerical imports or artifacts; then loader and full admission. Original
  launch time persists across resumes. Provisional shared cap is2h per teacher for
  all3 arms plus final validation, including imports/loading/downtime, or absolute
  2026-10-29T06:50:00Z, whichever earlier. Kernel SIGALRM enforces hard cap;
  timeout preserves last atomic checkpoint and never extends cap on resume.
- `FitSession` locks teacher hash/seed, data identity, replacement source identity,
  environment, Adam settings, architecture, three initialization hashes, shared
  schedule hash, teacher validation correct and original launch/plan hash.
  Initial snapshot file includes all3 arms. Exact calibration-ID batch identity
  checked before each update. Caller supplies actual reviewed replacement source
  hashes/commit as source identity; no invented approval hash.
- Atomic checksummed weights-only checkpoints include three models/optimizers,
  initial snapshots, Python/NumPy/Torch RNG, shared completed-update cursor and
  lock. Save after each complete three-arm update. Post-update finite guards cover
  parameters, gradients and optimizer state; partial/failed steps cannot save.
  Resume rejects changed teacher/recipe/data/source/environment/cap, invalid cursor,
  optimizer groups, initial hashes, payload corruption and nonfinite states.
- Exactly1000 updates -> fixed final checkpoint -> validation-only report, binding
  final checkpoint hash. No best selection. Validation checks approved exact IDs,
  length/order, no calibration overlap, and labels accessed only in evaluation.
- Cohort release gate only verifies a separately reviewed manifest binding all3
  planned statuses and complete recipes/checkpoint/validation hashes. It contains
  NO test evaluator/reader and cannot automatically release or decode test data.
  Default fit/loader guards reject invocation without explicit execution flag.

Synthetic tests cover architecture, initialization, BN frozen paths, local/joint
gradients, identity admission, centered batch-invariant NMSE, original cap, exact
resume, corrupt/nonfinite last-good preservation, validation gate, cohort gate,
restricted loading and no-test-member requests. These are implementation checks,
not wall-time feasibility or scientific outcomes. Parent will review and announce
final plan to the user before any actual student fit. No head/suffix/BN repair.

### Review follow-up: required trusted orchestration boundary

Reviewer found no scientific architecture/loss/split/NMSE blocker, but required
explicit runtime admission/resource integration. Added production AdmissionBundle
from full pinned admission, carrying teacher/data/seed/quality/hash/model-state
binding; fit rejects independent or mutated teacher/data objects. Added stdlib
`launch.bootstrap` that rejects prior numerical imports, takes a nonblocking single
replacement-worker flock, sets4-thread library environment, opens durable launch
and arms hard deadline. Caller retains worker-lock handle through fit, imports
numerical libraries then sets torch intra-op4/inter-op1 and deterministic mode.
Production fit verifies those settings and durable launch equality. This4-thread,
maximum1worker choice is PROVISIONAL pending parent's final announced plan.

This remains library plumbing, not a separately reviewed standalone executable.
The final trusted caller must prove plan approval, bootstrap ordering, held lock,
preliminary teacher quality before data access, source identity and complete
production admission. Typed bundles/booleans are accidental-misuse guards, not
security boundaries against malicious in-process code. No end-to-end real-data
execution was performed; no execution readiness is claimed solely from these tests.

Parent provisionally accepted4threads/max1 replacement worker, with a GLOBAL
8CPU-thread science limit across tasks. Parent schedules slots: no replacement
launch while two4-thread teachers are active. Wait for a free heavy-job slot;
no distributed scheduler or autonomous launch. Existing teacher work has priority.

## Minimal manifest-driven future entry (still NOT RUN)

`python -m research.cifar10_replacements.entry --plan PLAN.json
--plan-sha256 SHA256 --slot-manifest SLOT.json --slot-sha256 SHA256 --seed SEED`
is now a prepared entry, **not** an authorization. Optional
`--parent-approved-resume` also requires the current parent slot manifest to
explicitly authorize resume. Default refuses an existing run directory; completed
validation cannot be resumed. Parent still announces final plan before science.

Parent supplies immutable trusted hashes, not just paths. Plan schema requires:
`status="parent-approved-and-user-announced"`, `planned_seeds` containing all3
fixed seeds in order, `threads=4`, `interop_threads=1`,
`max_replacement_workers=1`, `global_science_threads=8`,
`per_teacher_cap_seconds=7200`, `absolute_deadline=1793256600.0`,
`official_test="sealed"`, `updates=1000`, approved `preflight` path,
`source_sha256` mapping every entry in `entry.SOURCE_FILES`, and `runs` mapping
planned decimal seed strings to output/teacher artifact locations and SHA256.
Each run requires `output`, `teacher_export`, `teacher_export_sha256`,
`teacher_metrics`, `teacher_metrics_sha256`, `teacher_checkpoint`,
`teacher_checkpoint_sha256`. Runs may be added through a separately approved
per-seed plan as teachers finish; already-started plan hashes never change on
resume. The intended cohort and common recipe remain fixed across these manifests.

Slot schema requires `status="parent-reserved-free-slot"`, bound `plan_sha256`,
`seed`, `output`, `replacement_threads=4`, integer `other_science_threads`0–4,
`global_science_threads=8`, explicit boolean `resume_authorized`, and future Unix
`expires_at`. It is a parent-reserved slot, not a worker's CPU observation.
Parent must retain the slot allocation throughout the process; entry has no
cross-task scheduler. All source hashes are checked before bootstrap.

Order: verify source/plan/slot -> exclusive output creation (or explicit resume)
-> stdlib timed bootstrap and retained single-worker lock -> hash-verified JSON
final teacher quality gate -> numerical imports/Torch4/1/deterministic -> pinned
training-only loader -> checkpoint hash/full admission bundle -> fit -> fixed
final validation. No test decoder or automatic cohort release is added.

Append-only `exit_<timestamp>.json` records planned seed, plan/slot/teacher hashes,
ordinary failure/exit status and last checkpoint hash where available; success
adds replacement/recipe/validation hashes. Parent must collect actual process
exit, especially hard SIGALRM/OS termination, which cannot guarantee a receipt.
Original2h/global caps remain unchanged across explicit resumes. No automatic
retry, seed substitution, best selection, tuning or repair.
