> 歴史的な事前計画の科学部分を抜粋した資料。本文の準備・未実行という時制は当時のもの。現在の実行状態は結果報告を参照。

> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Fixed spatial-sampling contrast — source preparation, NOT GO

This is a **new, adaptive, descriptive-only mechanism experiment**, not an extension,
restart, or reanalysis of the completed utility-gap clock. This preparation does
not authorize fitting, reporting, data access, publication, or additional variants.
The inherited original pilot, utility-gap, validation and official-test negative
evidence remains unchanged. The official test is unblinded and is not reusable
as a confirmation resource. The source proposal is preserved in
PROPOSAL_CONTEXT.md（元資料のみ・この抜粋には未同梱）.

## One hypothesis, six fresh trajectories

Exactly seeds **26102601, 26102602, 26102603**, with the same admitted fixed-final
epoch120 teachers; no retraining/substitution. Exactly two fresh rank8, two-map
MSE replacements per teacher:

| Arm | 3×3 dilations/padding | Maximum potential span-input support | Weights | Nominal replacement conv MACs/image |
|---|---|---|---:|---:|
| rf5 control | (1,1) / (1,1) | 5×5 | 5,120 | 1,310,720 |
| rf9 structural | (1,3) / (1,3) | dense 9×9 | 5,120 | 1,310,720 |

Each map is `ReLU(x + Conv1x1(8→32)(ReLU(Conv3x3(32→8)(x))))`, bias-free,
without BN. Both have two residual additions, four ReLUs and output32×16×16.
Offsets per axis are `{-1,0,1}+{-3,0,3}={-4,…,4}`. This is **potential support
relative to the span input**, not raw-image RF, effective RF, or equivalence to
the teacher's four dense convolutions/BN/nonlinear paths. Dilation also changes
sampling geometry, conditioning and boundary behavior, so a positive contrast
cannot uniquely identify RF size as the cause.

Control initialization is the fixed seed/default Conv2d initialization, created
under an isolated Torch RNG. Structural tensors are exact independent copies;
initial functions differ because dilation differs. Every seed/arm initialization
hash, schedule hash, source dependency hash, input identity and environment is
frozen before the first fit. Worker rechecks all six initial identities/schedules.

Only endpoint elementwise-mean MSE; Adam lr0.001, betas(0.9,0.999), eps1e-8,
weight_decay0, fresh Adam, no clipping/augmentation/learning-rate changes. Exactly
64 immutable passes through fit4,000: batch128 and final32, 32 updates/pass,
**2,048 updates per trajectory; 12,288 total; 1,536,000 presentations** if all
complete. Both arms receive the same exact per-seed order. The teacher, suffix,
classifier and all BN buffers remain EVAL/frozen, verified by full state hashes.
No labels or diagnostic capability exists in the fitting path. Loss, gradient
norm and parameter-update norm are audit logs only, never adaptation signals.

## Seen-data boundary and outcome definition

Reuse only the existing integrity-bound calibration-only cache produced under
the reviewed gap ingestion protocol. **No new archive ingestion is provided.**
Reuse the exact previously frozen4,000/1,000 partition and teachertrain
normalization. Diagnostic images/outcomes are already seen; earlier pilot
students used these images for fitting. Contemporaneous fresh controls are not
independent replications. No original-validation or official-test arrays are
loaded, evaluated, repartitioned or used here.

Only final update2,048 is an outcome; no1,024 measurement, best checkpoint,
substitution, warm-start, resume, retries, extra seeds or branch search. All six
checkpoint/failure statuses freeze before **one** scheduled reporting phase.
Both fit and diagnostic regional metrics are computed in this phase, not during
training. Final-fit metrics require24,000 student-image passes and diagnostics
6,000 plus teacher passes; runtime is unmeasured.

Primary: centered teacher-endpoint NMSE on the fixed central spatial locations
`[4:12,4:12]`, independently for fit and diagnostic. Sample-axis centering occurs
at each channel/location. SSE / (`centered_teacher_SS + 1e-12*N*32*locations`),
with locations64 interior,256 full,192 border; every region uses its own
statistics/denominator. Interior has no direct zero-padding exposure under
either support. Full-map and border are fixed secondary measures, never fallback
endpoints. Accuracies, exact teacher drops, paired error and prediction
counts are secondary descriptive results. All three teacher and all six student
prediction/status records, finite logits, regional metrics, and unchanged
teacher hashes freeze durably **before** the one label opening. Exact labels,
logits, predictions and sufficient statistics persist for read-only audit.

Prediction: rf9 reduces **both** interior fit and interior diagnostic NMSE by
at least10% in **every** teacher, with no worsening of full-map diagnostic NMSE
in any teacher. Report per-teacher contrasts and all-three conjunction. Six
complete trajectories with a failed conjunction mean the specified prediction
failed under this architecture/recipe. Mixed results remain mixed; missing,
capped, nonfinite or failed reporting cells mean incomplete/inconclusive, not
zero effects. A zero reference makes relative reduction undefined, not success.
No p-value, population inference, formal noninferiority, utility pass or fresh
confirmation claim follows. Even≤2pp observed secondary teacher drop cannot
revise earlier negative results. No outcome authorizes another experiment.

## New immutable clocks, capacity and one-look custody

- New overall cap **23,400s (6.5h)**, no later than **2026-10-29 06:50 UTC**.
  Latest full-allocation start is2026-10-29 00:20 UTC. Entire6.5h live capacity
  reservation required before launch, one4-thread worker, inter-op1, total
  science threads≤8; maximum26 CPU-thread-hours. The existing shared worker
  lease prevents concurrent replacement workers.
- Every teacher gets at most7,200s for both fits including imports/loading and
  downtime. Start only if its full2h fits before the overall reporting reserve.
  overhead are not a guarantee of completion; missing allocations remain missing.
- One reporting allocation at most1,800s, within the same6.5h/global deadline.
  `cohort_started.json` consumes this experiment identity before launch, so a
  different/copied output cannot restart six more fits. After the full-six
  are rechecked before numeric imports. Exclusive phase worker markers prevent
  direct child replay. The supervisor installs a hard owned process deadline
  before numerical imports/payload access and reaps owned children on errors.
  Generic reader/OS timeouts are failed, not cap evidence; only owned deadline
  signals or request-bound typed exit evidence establish cap exhaustion.
- All six statuses persist after handled controller failures. Uncertain process
  liveness blocks another worker and reporting. Unverified/partial checkpoint
  files never become completed cells. A partial optimizer exception reports
  unknown update count; an established applied update is counted even if later
  logging/finiteness fails. Missing/checksum-corrupt receipts preserve explicit
  unknown counts rather than invented zeros. Raw failure evidence is retained.

## Reuse and limits of the implementation

New code lives only under `research/cifar10_spatial_contrast`; inherited files
remain byte-identical to reviewed runtime **bc7d53a984f8f3a553b759013eb2f66d4db78da9**.
Reuse is explicit: old pure partition/hash helpers; default optimizer and immutable
schedule; teacher wrapper/hash/finite helpers; EndpointStats; calibration-only
manifest/array validators and image capability; pinned original-teacher
admission; paired counts; atomic checkpoint bytes. The new minimal process
boundary adapts the reviewed lease/hard-deadline/reaping logic, changing module
identity. It does not invoke old fit/diagnostic launchers or their global ledger.

resource stewardship remain necessary. Python objects are not a security sandbox
against a malicious in-process caller. Metadata assertions do not prove new
runtime feasibility. No scientific fit, actual image/label evaluation, existing
model deserialization, network or CI occurred during this preparation.

See LAUNCH_PREREQUISITES.md（元資料のみ・この抜粋には未同梱） for exact admission steps.
