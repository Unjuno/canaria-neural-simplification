# PROVISIONAL Stage A: fixed-final CIFAR-10 teacher

**Preparation only. No experiment has started.** Pending completed preflight
feasibility and explicit parent-announced execution plan. Source base 9154339.
No autostart, download, CI, push, fees or user-PC execution. This protocol does
not authorize invocation. Only synthetic-array unit tests may run now.

Provisional independent seeds: 26102601, 26102602, 26102603. `git grep` of
9154339 found no prior occurrences of these decimal seeds (not proof against
an equivalent differently encoded seed). No seed selection on performance.

CPU float32, four intra-op/one inter-op threads, deterministic PyTorch algorithms.
OMP/MKL/OpenBLAS/NumExpr thread environment pinned to four before imports.
Python3.13.5 / torch2.10.0+cpu / numpy2.3.5 are required by the CLI.
Source hash and dependency/platform/version/thread snapshot are checkpoint-bound; cross-environment
resume is rejected. No claim of bitwise portability to other CPUs/libraries.

Exact preflight ResNet20 initialization and option-A architecture (269722 params).
Read ONLY the five official training batch members from the existing checksum-
verified archive. Never open/decode the official test member. Archive hashing and
gzip stream traversal necessarily cover its compressed bytes. Require preflight
complete metrics and exact manifest member hashes. Recreate PCG64 split20261026:
4000 train/500 calibration/500 validation per class. Check saved index identity.
Recompute and verify normalization from teachertrain only. Calibration is excluded
from both training and evaluation. Read-only preflight cache; separate output.

Exactly 120 epochs, shuffled without replacement, retain final partial batch
(40000 => 313 batches/epoch), batch128, zero pad4/random crop32/flip p=.5.
SGD momentum .9, weight decay 5e-4; lr .1 for zero-based epochs0–59,
.01 for60–89, .001 for90–119. No other schedule, early stopping, validation peeks,
or best-checkpoint selection. Final epoch120 unaugmented validation only;
teacher adequacy >=85% on fixed validation, reported per seed. Inadequacy is a
result, not permission to retune/restart/select. No official-test evaluation.

Atomic SHA256-enveloped checkpoints after each completed epoch and every five
minutes at completed minibatch boundaries, plus final checkpoint. Persist model,
optimizer, Python/global NumPy/PyTorch RNG, PCG64 augmentation/shuffle RNG, current
permutation/cursor, completed epochs, train indices, normalization and binding.
Checkpoint schema2 uses tensors and primitive structures only; NumPy RNG keys
and permutations are lists reconstructed under a controlled schema. Loading uses
`torch.load(weights_only=True)` exclusively, with no custom-global allowlist or
unsafe fallback. Treat artifact origins/integrity as untrusted until verified.
SIGTERM/SIGINT pause at next batch boundary and checkpoint. Failure mid-batch
retains last good checkpoint, never serializes partially applied updates.

10h per-seed wall-clock cap measured from the early stdlib bootstrap timestamp,
including numerical-library imports, data loading and downtime across resumes.
An exclusive durable `launch.json` is created once before numerical imports;
resume reads that original timestamp and arms its original absolute expiry
before numerical imports or any dataset access. Expired resume is rejected
without decoding data. Checkpoint timestamp must agree with the launch record. Absolute research deadline
2026-10-29T06:50:00Z. Checks at batch/validation boundaries plus kernel-default SIGALRM hard termination
at the earlier cap. Hard termination retains the last atomic checkpoint; it
never saves partially applied native operations. No subsequent batch may start
once a cap is reached. Cap is not
an instruction to use all remaining time. No automatic launch of other seeds.

Future explicit invocation from worktree root (NOT RUN during preparation):
```
/workspace/shared/canaria_r87r2_training_20261026/.venv-locked/bin/python -m research.cifar10_teacher.run --execute-provisional-plan \
  --preflight-artifacts /path/to/completed/preflight/artifacts \
  --output /separate/path/seed26102601 --seed 26102601
```
Resume uses identical arguments plus `--resume`. Final validation does not begin
until checkpoint epoch120 and is never used for model selection. Metrics records
final checkpoint hash, environment, elapsed wall time and data-manifest hash.

## Portable evaluator identity

Every checkpoint embeds explicit model configuration and its canonical sorted-JSON
SHA256, split-index SHA256 values (little-endian int64), and teachertrain
normalization values plus SHA256 (mean then std, little-endian float64).
The final checkpoint additionally exports `teacher_export.json`, a plain JSON
identity manifest linking its full-file SHA256, data-manifest SHA256, seed, epoch,
model configuration, all three split identities, normalization, and environment.
This does not authorize any comparison or evaluation. Consumers must also require
completed `metrics.json` and its adequacy gate; export presence alone means the
fixed checkpoint exists, not that final validation succeeded or passed.

Post-update finite checks cover every model parameter/buffer and optimizer tensor/scalar
before advancing the cursor. Checkpoint save repeats these checks. Overflow never
overwrites the last good checkpoint. Schema1 preparation checkpoints are not accepted.
