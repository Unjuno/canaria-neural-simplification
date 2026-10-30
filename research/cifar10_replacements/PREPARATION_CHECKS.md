# Synthetic-only preparation evidence

2026-10-26, worktree `/workspace/shared/canaria_cifar_replacements_20261026`,
branch `research/cifar10-replacement-prep-20261026`, base `67bd076`.

Locked interpreter:
`/workspace/shared/canaria_r87r2_training_20261026/.venv-locked/bin/python`.
OMP/MKL/OpenBLAS/NumExpr and PyTorch intra-op threads limited to1.
Five tests passed in1.018s on first invocation (09:32UTC):

- Exact5120 parameters each; output shape; no BN/bias; explicit ReLU formula;
  bitwise Sep initial parameter identity with independent storage and optimizer state.
- Frozen teacher/no-BN-mutation; expected endpoints; original span installed gives
  exact original logits. Only synthetic images were forwarded.
- Local teacher forcing versus joint predicted intermediate; detached targets;
  exact standalone local gradient scale; both local maps update.
- Identical1000-update batch schedule; all5000 IDs once per40 updates, final8;
  common aggregate NMSE and invalid denominator handling; no filesystem/data helper.
- Synthetic checkpoint acceptance at85%; rejection below gate, nonfinal epoch,
  failed/pending validation, seed/data/config/file hash mismatch and split leakage flags.
  Admission succeeds with filesystem opening mocked to fail.

No real data or teacher artifact access, training/evaluation, network access,
download, push, CI, charge, or formal-teacher interference. Tiny synthetic forward,
backward and optimizer steps are implementation checks, not CPU science.

After adding complete expected-identity field/split-size/normalization-source
admission guards, all five tests passed again in1.064s at09:34UTC. `git diff
--check` passed. Approved identity contents still require independently trusted
preflight reconstruction by future production loader, as documented in protocol.

## Minimum execution-plumbing extension — PREPARATION ONLY

2026-10-26 09:50UTC. Added train-only loader pinned to source manifests67bd076/
cfcc082 (source text only), production admission bundle, stdlib bootstrap, fixed
three-arm fit/checkpoint/resume plumbing and validation-only/cohort-release guards.
No real archive, data partitions, teacher checkpoint, teacher metrics or outcomes
were read. No actual fit or evaluation was launched.

All18 tiny synthetic tests passed in1.450s using existing locked interpreter and
one numerical thread; `git diff --check` passed. Coverage includes prior checks
plus source/spec admission, exact split membership, restricted pickle/no-test
member extraction, centered NMSE across fullN and batch aggregation, exact model
and Python/NumPy/Torch RNG resume, original cap/expired resume before artifact read,
finite post-update failure preserving last good snapshot, fixed-final validation
including synthetic successful report, separate cohort gate, and production
bootstrap/admission guard rejection. Fake archive arrays use broadcast backing;
synthetic validation uses5000x32x1x1 identity-map features, not real-image science.

One intermediate17-test run caught two weights-only resume failures caused by
serializing torch's TorchVersion object. Changed environment snapshot to plain
`str(torch.__version__)`, retaining weights_only=True without unsafe fallback.
Subsequent17 tests passed1.485s and final18 passed1.450s. No scientific results
were available or used for these changes.

Reviewer confirmed centered per-feature fullN NMSE definition and found no
architecture/loss/data-split scientific blocker. Integration recommendations
addressed with typed admitted-teacher binding, durable bootstrap/thread check,
single-worker lock and explicit trusted-caller boundary. Parent controls global
8thread slots; proposed own4threads/max1worker is provisional. Full actual-data
orchestration remains NOT RUN and requires parent-announced final plan, teacher
quality gate, reviewed caller and free science slot. No test evaluator exists.

## Minimal future-entry preparation

Added stdlib manifest/source/free-slot verification and guarded entry, retaining
bootstrap lock through approved data/admission/fit library sequence. Three mocked
entry tests passed in0.006s at09:55UTC: exact gate/import/admission/fit order,
failed teacher admission before numerical/data work, plan hash/slot/resume guards,
no default overwrite and success/failure receipts. No numerical science functions
were invoked: synthetic JSON/bytes, temporary directories and mocked services only.
Actual entry remains NOT RUN pending parent announcement, final adequate teacher,
valid reserved CPU slot and final reviewer source check.
