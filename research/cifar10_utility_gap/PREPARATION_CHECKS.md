# Source preparation checks

**58 synthetic tests PASS** (23 initial new +14 review-regression +21 existing replacement regression), **18 repository tests PASS**, repository audit PASS and publication integrity audit PASS. The existing announcement block `BLOCKED_REPRODUCTION_ISSUE_87` is unchanged.

Commands, run locally from the isolated worktree, with no network or CI:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH=src:.
PY=/workspace/shared/canaria_r87r2_training_20261026/.venv-locked/bin/python
$PY -m unittest research.cifar10_utility_gap.test_synthetic \
  research.cifar10_utility_gap.test_review_regressions \
  research.cifar10_replacements.test_synthetic \
  research.cifar10_replacements.test_data_synthetic \
  research.cifar10_replacements.test_runtime_synthetic \
  research.cifar10_replacements.test_entry_synthetic -v
$PY -m unittest discover -s tests -v
$PY tools/audit_repo.py
$PY tools/audit_publication.py
```

New tests assert1 intra-op/1 inter-op thread. Tiny synthetic teacher forward/backward and one optimizer step validate the differentiable KL suffix; no actual teacher artifact or CIFAR pixels were loaded. Synthetic zero-valued array payloads validate capability separation. Existing regression tests use synthetic fixtures/mocked orchestration only. No long trajectory or real-data benchmark was run. Checkpoint boundary tests deliberately construct synthetic log/Adam/counter consistency fixtures; they are not scientific fits.

Coverage: approved partition/source/original-cohort/slot bindings; denial before payload callback; exact SHA256 partition and64-pass schedule; model parameter counts; isolated initialization/equal objective copies; fresh Adam; T=1 KL; nonzero replacement gradients; detached targets; no teacher gradients/BN changes; centered NMSE/sample weighting; finite last-good atomic checkpoint, Adam/RNG serialization; exact fixed budgets and missing-cell retention; no warm-start/resume/continuation; all24-checkpoint freeze; exclusive one-look diagnostic image/label markers; persisted prediction freeze before labels; exact paired counts; descriptive flags, thresholds/zero baseline/missing cells; no dataset decoder/network/validation API.

`SOURCE_FREEZE.json` binds reviewed source/dependencies. `PREPARATION_EVIDENCE.json` binds logs and distinguishes preparation from scientific execution. Independent review remains pending. No source/manifest gate or passing synthetic test is execution approval.


Independent review repairs: initial probe outputs and pre-repair negative-test failure receipt are retained in `evidence/review_findings/`. The seven initial negative tests had nine expected failing assertions before repair. Fourteen review-regression tests now exercise the repaired guards, including valid bound checkpoint/metric acceptance, rejection of log/Adam/initial/RNG inconsistency, exact one-use metric receipts, malformed/foreign metric bytes, terminal-state stability, exception-versus-timeout accounting, integer classes and full-allocation slot admission. Latest complete logs are `evidence/review_findings/synthetic_tests_after_repair.log`, `repository_tests_after_repair.log`, and both `audit_*_after_repair.log`; original green preparation logs remain unchanged.

The exact pre-repair unittest receipt contains two trailing spaces emitted on failed subtest lines. Raw `git diff --check` flags those two evidence-log lines; they are retained byte-for-byte rather than rewriting the failure receipt. The source/JSON/Markdown diff check (excluding raw `.log` receipts) passes.


The later generic-reader-timeout red receipt is also preserved (`reader_timeout_before_repair.log` plus its exit receipt). Dedicated deadline-expiry and unrelated reader/update timeout tests distinguish cap exhaustion from ordinary failures. Latest complete run: `evidence/review_findings/synthetic_tests_after_deadline_fix.log`.
