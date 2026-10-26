# Final independent preparation review — b7cbd1b

## Verdict

**GO for source preparation at `b7cbd1b906e7a46cf8a6ed651b1cfa1bd5064a75`. SCIENTIFIC EXECUTION: NOT GO.**

The original six guard defects and the subsequent generic-timeout accounting defect are repaired and independently verified. This is approval of the bounded source-preparation deliverable and its synthetic checks, **not** approval to ingest data, load a real teacher, launch fitting, open diagnostic payloads, or run official-test evaluation.

## Independent verification

All reruns used the exact archived b7cbd1b source snapshot, one intra-op/one inter-op thread, the existing locked Python interpreter, and only synthetic arrays, randomly initialized tiny-network exercises, constructed audit fixtures and mocked failures.

| Check | Result | Evidence |
|---|---|---|
| Utility-gap original and repair suites | **37/37 PASS**, 2.230 s | `synthetic_tests_b7cbd1b.log` |
| Independently authored original-negative / mutation / timeout probes | **8/8 PASS**, 0.841 s | `recheck_a9af0b0.py`, `recheck_b7cbd1b.log` |
| Frozen source/dependency bytes | **15/15 hashes match** | `hash_validation_b7cbd1b.json` |
| Frozen evidence bytes | **24/24 hashes match** | `hash_validation_b7cbd1b.json` |

The independent probe file is deliberately unchanged from the a9af0b0 review: its last test failed there and passes here. The original a58d6ea and a9af0b0 failure receipts remain preserved, as do the initial NO GO reviews. The preparer's broader 58-test claim is recorded in its evidence; this final independent rerun is specifically the 37 utility-gap tests plus eight independent tests above.

## Verified repaired invariants

- Incomplete metric dictionaries cannot create a checkpoint. Required finite metric fields and centered-NMSE sufficient-statistic arithmetic are checked, with exact session/cell/update/model-state/fit-ID/teacher bindings and an internally generated, one-use receipt.
- Diagnostic admission verifies actual canonical full-fit metric bytes and their hashes/bindings. A hash assertion without the corresponding bytes is insufficient.
- Asserted update counters cannot bypass missing per-update logs or Adam state. Default initialization, log order/sample weighting, optimizer recipe/step/tensor census, parameter shapes/dtypes, nonnegative Adam second moments and RNG snapshots are checked. Independent mutations are rejected.
- Terminal receipts are stable copies; new fixed-checkpoint freeze and full-fit metrics are rejected after closure. Metric and checkpoint-write failures latch explicit failures and block retries. Previously persisted last-finite audit artifacts are not relabeled final checkpoints.
- `DeadlineExceeded` distinguishes the owned immutable deadline from ordinary `TimeoutError`. The previously failing reader timeout with 7,199 seconds remaining now produces **failed**; generic update timeout is also failed, while the owned-expiry test remains **cap-exhausted**.
- Nonfinite/boolean clock values, nonintegral class IDs and a reservation shorter than the next full two-hour teacher allocation are rejected.

The previously verified design/dataflow remains unchanged: full two-width × two-objective factorial design; 1,024/2,048 checkpoints within each fresh trajectory; 4,000-fit/1,000-diagnostic separation; equal objective initializations within a width; nonzero replacement gradients through a frozen suffix; detached teacher targets and immutable EVAL teacher/BN state; exact shared schedule and all-cell failure census; default-deny execution/release gates and durable single-use diagnostic markers.

## Explicit limits and required next gate

The caps remain **2 h per teacher fit+full-fit metrics, 30 min diagnostics, 6 h overall, and 2026-10-29 06:50 UTC globally**. No runtime feasibility claim follows from tiny synthetic tests. Constructed 1,024/2,048 counters/logs/Adam fixtures are consistency tests, not measured trajectories or proof that an untrusted producer optimized anything.

Scientific execution still requires a separate reviewed package binding:

1. The completed immutable original cohort, quality-admitted teachers and published real calibration partition.
2. An authorized, integrity-bound calibration-only producer/cache, or a separately approved narrow ingestion amendment. The old mixed-batch decoder does not satisfy the current strict boundary.
3. A stdlib bootstrap that validates actual source/approval bindings, reserves live resources and the shared worker lock, records clocks before numerical imports/loading, and installs/maintains the immutable hard deadlines. Lifetime reservation validity, process interruption/persistence outcomes and no automatic retry require end-to-end orchestration tests.
4. Genuine per-step/artifact provenance and complete checkpoint/metric/status freezing, followed only by separately approved, one-look private diagnostics with trusted directory custody and prediction-before-label release.
5. User-announced execution authority. No original-validation/test decoding, CI, network writes or fees are authorized by this review.

No real CIFAR payload or model artifact was loaded, no real fit/validation/test decode occurred, no network write or CI ran, and the reviewer made no source edits. The unresolved launch action remains **not attempted**, not a failed or uncertain running job.
