# Announcement readiness gate

This is the active pre-announcement checklist for Canaria. Historical v0.2.0 snapshot/release records are archived under `../archives/releases/v0.2.0/` and are **not** the current gate.

**Current status: NOT READY FOR BROAD ANNOUNCEMENT.**

Tracking issue: #13.

## A. Scientific claim surface

- [x] Core wording is operational replacement/description complexity, not Kolmogorov or universal mathematical complexity.
- [x] Phase 2E is explicitly `INVALIDATED_IMPLEMENTATION_BUG` / `DO_NOT_USE_FOR_INFERENCE`.
- [x] Phase 2O positive repair-sample-complexity claim is removed.
- [x] Training-time primary, secondary, negative, and exploratory evidence classes are separated.
- [ ] Candidate post-snapshot external-validity evidence has received an explicit inclusion/exclusion review for the announcement claim set.
- [ ] Issue #15 stronger-teacher regression control is completed or explicitly waived with the weak-teacher limitation retained.
- [ ] Final claim registry has been reread after all candidate evidence decisions.

## B. Headline reproducibility

- [x] Representative residual-MLP seed-1200 runner is clean of private `/mnt/data` dependencies.
- [x] Historical seed-1200 smoke endpoint is `component-wise=3072`, `composed=1536`.
- [x] A pinned Python 3.11 reproduction environment is recorded for the current headline runner.
- [ ] Fresh seeds `1200–1207` have been regenerated from a clean checkout under the pinned environment.
- [ ] Regenerated minimum passing budgets match the committed per-seed primary endpoints.
- [ ] Aggregate primary statistics reconstructed from regenerated outputs match the committed summary within declared numerical tolerance.
- [ ] A durable reproduction report identifies Python/package versions, commit, and outcome.

The first one-shot full-cohort Actions attempt did not yield usable reproduction evidence; this item remains open until a clean completed run is obtained and checked in. Infrastructure failure is not a scientific failure.

## C. Repository integrity and readability

- [x] Current evidence is separated from the older version-number research sequence on the Issue #16 restructure branch.
- [x] Historical blobs/results are preserved under `archives/research-history/` rather than deleted.
- [x] Historical release/review gates are separated from current readiness documents.
- [x] Public/reproduction runners are checked for private `/mnt/data` dependencies.
- [x] `tools/audit_repo.py` guards Phase 2E invalidation and selected semantic invariants.
- [ ] Issue #16 restructure has passed link/integrity audit and review.
- [ ] `python -m unittest discover -s tests -v` passes on the final candidate commit.
- [ ] `python tools/audit_repo.py` passes on the final candidate commit.
- [ ] `python tools/audit_readiness.py` passes on the final candidate commit.
- [ ] GitHub `repository-audit` passes on the final candidate commit.
- [ ] No one-shot experiment/maintenance workflows remain on the final branch tip unless intentionally supported.

## D. Communication and metadata

- [x] README and STATUS explicitly say the repository is pre-announcement.
- [ ] `docs/CLAIMS_AND_EVIDENCE.md` matches the final included evidence set.
- [ ] `docs/PUBLICATION_NOTES.md` matches the final included evidence set and remaining caveats.
- [ ] `CITATION.cff` and release metadata do not imply a stronger readiness state than the repository actually has.
- [x] Historical `v0.2.0-public-snapshot` is described as a frozen research snapshot, not a current announcement-ready release.
- [ ] No README badge, release title, or top-level wording implies production/library maturity that the project does not have.

## E. Exit rule

Announcement readiness requires every unchecked item above to be either completed or explicitly waived in Issue #13 with a documented reason and corresponding limitation in the final claim surface.

A passing experiment alone does not close this gate. A passing CI run alone does not close this gate. Close Issue #13 only after a final integrated review of science, reproducibility, repository structure, and communication surfaces.
