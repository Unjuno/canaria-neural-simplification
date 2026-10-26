# Preparation evidence (not scientific experiment results)

2026-10-26, source base 9154339; separate worktree/branch
`research/cifar10-teacher-stage-a-20261026`.

- No occurrences of 26102601/26102602/26102603 found by `git grep -n -E`
  against base9154339. Seeds remain provisional.
- Default container Python lacks torch: first unittest invocation failed during
  import, before computation. No installation was attempted.
- Existing locked interpreter:
  `/workspace/shared/canaria_r87r2_training_20261026/.venv-locked/bin/python`
  Python3.13.5, torch2.10.0+cpu, NumPy2.3.5.
- `python -m unittest research.cifar10_teacher.test_synthetic -v`:
  five tests passed, 1.070s, completed 2026-10-26T09:05:09Z.
  Earlier pre-hardening invocation also passed (five tests, 1.001s).
- Tests: exact pinned architecture source/269722 parameter/shape match;
  synthetic stratification and train-only normalization invariance; bitwise
  optimizer/model/RNG/permutation/cursor resume mid-epoch and at epoch boundary;
  checkpoint digest corruption and data-binding rejection; LR boundaries and
  final-only validation/train-overlap guards. No validation accuracy computed.
- `git diff --check` and source compilation passed.
- Only tiny synthetic arrays were used. No real CIFAR data was accessed,
  trained on or evaluated; no network/download/CI/push/fee/user-PC action.
  Active preflight source and cache were not changed.

Readiness is implementation-level only. Pending complete data/runtime preflight,
parent feasibility decision and announced execution plan. Quality and projected
wall-time remain unestablished by these preparation tests. Future production
loading must still verify the preflight manifest, archive and exact partitions.

## Portable evaluator identity follow-up

At parent request, added explicit architecture configuration/canonical JSON hash,
all three split index hashes, normalization values/bytes hash to checkpoints,
and a final plain-JSON `teacher_export.json` identity manifest. No comparison
implementation or execution. Export requires separate complete adequacy metrics
before downstream admission. Six synthetic-only tests passed in0.970s at
2026-10-26T09:14:21Z, including portable hash encoding checks and exact resume.
Preflight worker confirmed download-only/below130MiB before these brief tests.

## Static-review corrections (tests pending quiet window)

Addressed original resume deadline bootstrap, post-update model/buffer/optimizer
finite protection, and restricted weights-only checkpoint loading. Added tests
for expired resume rejection before numerical imports, unsupported object rejection,
and preserving last-good checkpoints after parameter/buffer/momentum corruption.
These new tests have NOT yet run: preflight benchmark has priority; no CPU tests
will run until parent/worker confirms its timing window has ended.

Preflight worker confirmed timing ended2026-10-26T09:23:38Z. Subsequently all
NINE synthetic-only tests passed in1.265s, completed09:24:26Z, including all
three new blocker regressions. No real dataset access or formal training.
