# Announcement readiness gate

This is the active pre-announcement checklist for Canaria. It is deliberately separate from `RELEASE_CHECKLIST.md`, which records the historical v0.2.0 snapshot/version-control sequence.

**Current status: NOT READY FOR BROAD ANNOUNCEMENT.**

Tracking issue: #13.

## A. Scientific claim surface

- [x] Core wording is operational replacement/description complexity, not Kolmogorov or universal mathematical complexity.
- [x] Phase 2E is explicitly `INVALIDATED_IMPLEMENTATION_BUG` / `DO_NOT_USE_FOR_INFERENCE`.
- [x] Phase 2O positive repair-sample-complexity claim is removed.
- [x] Training-time primary, secondary, negative, and exploratory evidence classes are separated.
- [ ] Candidate post-snapshot external-validity evidence has received an explicit inclusion/exclusion review for the announcement claim set.
- [ ] If stronger-teacher regression evidence is judged necessary, a new locked protocol has been completed without modifying the finished Phase 3 protocol.
- [ ] Final claim registry has been reread after all candidate evidence decisions, not merely inherited from an earlier snapshot review.

## B. Headline reproducibility

- [x] Representative residual-MLP seed-1200 runner is clean of private `/mnt/data` dependencies.
- [x] The historical seed-1200 CI smoke endpoint is `component-wise=3072`, `composed=1536`.
- [x] A pinned Python 3.11 reproduction environment is recorded for the current headline runner.
- [ ] Fresh seeds `1200–1207` have been regenerated from a clean checkout under the pinned environment.
- [ ] Regenerated minimum passing budgets match the committed per-seed primary endpoints.
- [ ] Aggregate primary statistics reconstructed from regenerated outputs match the committed summary within declared numerical tolerance.
- [ ] A durable reproduction report identifies Python/package versions, commit, and outcome.

## C. Repository integrity

- [x] Historical evidence paths are preserved instead of mass-renamed.
- [x] Public/reproduction runners are checked for private `/mnt/data` dependencies.
- [x] `tools/audit_repo.py` guards the Phase 2E invalidation and other selected semantic invariants.
- [ ] `python -m unittest discover -s tests -v` passes on the final candidate commit.
- [ ] `python tools/audit_repo.py` passes on the final candidate commit.
- [ ] GitHub `repository-audit` passes on the final candidate commit.
- [ ] No one-shot experiment/maintenance workflows remain on the final branch tip unless intentionally supported.

## D. Communication and metadata

- [x] README and STATUS explicitly say the repository is pre-announcement.
- [ ] `docs/CLAIMS_AND_EVIDENCE.md` matches the final included evidence set.
- [ ] `docs/PUBLICATION_NOTES.md` matches the final included evidence set and remaining caveats.
- [ ] `CITATION.cff` and release metadata do not imply a stronger readiness state than the repository actually has.
- [ ] Historical `v0.2.0-public-snapshot` is clearly described as a frozen research snapshot, not a current announcement-ready release.
- [ ] No README badge, release title, or top-level wording implies production/library maturity that the project does not have.

## E. Exit rule

Announcement readiness requires all unchecked items above to be either:

1. completed; or
2. explicitly waived in Issue #13 with a documented reason and a corresponding limitation in the public claim surface.

A passing experiment alone does not close this gate. A passing CI run alone does not close this gate. Close Issue #13 only after a final integrated review of science, reproducibility, and communication surfaces.