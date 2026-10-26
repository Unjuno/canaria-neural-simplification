# Source/synthetic receipt — 2026-10-26

Base reviewed replacement source: `54b4c1852481db3f809913d5b048e046e363a0b6`.
Prepared in separate branch/worktree `research/cifar10-heldout-prep-20261026`.

Command (one intra-op and one inter-op CPU thread; no actual data/evaluation):
```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/workspace/shared/canaria_r87r2_training_20261026/.venv-locked/bin/python \
-m unittest research.cifar10_heldout.test_synthetic -v
```
Seven tests PASS: pure valid/invalid release guards with all file opens denied,
source mismatch order, synthetic-byte hash mismatch, rejected admission before
mock test loader, exact paired counts and partition-invariant full-N centered
NMSE, restricted unpickler forbidden global rejection. Tiny fabricated tensors
only. No actual CIFAR data, test archive, teacher/student checkpoints read or
deserialized. No download/network, CI, push, charges, or real fit/evaluation.

This is NOT an independent review, provenance attestation, release or authority
to execute. Full artifact compatibility remains untested until final approved
artifacts exist; no claim is made that an end-to-end real run succeeded.

## Release-guard fixes after independent NO-GO

Ten tests now PASS (2026-10-26; same single-thread command; 2.249s test time).
New checks reconstruct expected schedule from approved calibration IDs/seed,
initial model hashes from fixed constructors, and Adam parameter groups/state
coverage with every parameter at step1000. Synthetic altered schedule, jointly
altered initial snapshot/hash, learning rate, step999 and missing optimizer state
all reject before mocked test loading. No optimizer updates are executed.
Registry tests verify file-fsync then parent-directory-fsync order, existing and
empty/partial registry rejection for different output paths, and fail-closed
parent-fsync errors before bootstrap. All registry files are temporary synthetic
fixtures; the real global registry was neither read nor created.
