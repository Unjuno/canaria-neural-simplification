# Canaria repository operations — integrated checkpoint navigation

> Updated 2026-10-31 (JST) for owner-authorized integration. Repository integration is not a new scientific result, independent replication, external peer review or release certificate.

## Choose the correct Git reference

| Git reference | Role |
|---|---|
| [`main`](https://github.com/Unjuno/canaria-neural-simplification/tree/main) | Integrated October research collection and navigation through [PR #108](https://github.com/Unjuno/canaria-neural-simplification/pull/108), together with preserved historical evidence and correction notices. |
| [`v0.2.0-public-snapshot`](https://github.com/Unjuno/canaria-neural-simplification/tree/v0.2.0-public-snapshot) | Immutable historical snapshot; never retag. |
| [`research/checkpoint-20261026`](https://github.com/Unjuno/canaria-neural-simplification/tree/research/checkpoint-20261026) | Source research checkpoint; documentation PR #106 and navigation PR #107 were merged here first. Retained for provenance. |
| [`docs/checkpoint-handoff-20261030`](https://github.com/Unjuno/canaria-neural-simplification/tree/docs/checkpoint-handoff-20261030) | Completed 14-document handoff, [merged PR #106](https://github.com/Unjuno/canaria-neural-simplification/pull/106); no longer an awaiting-merge proposal. |
| [`maintenance/repository-navigation-20261031`](https://github.com/Unjuno/canaria-neural-simplification/tree/maintenance/repository-navigation-20261031) | Completed navigation change, [merged PR #107](https://github.com/Unjuno/canaria-neural-simplification/pull/107); retained as a traceable source. |
| [`review/post-v02-independent-2026-09-12`](https://github.com/Unjuno/canaria-neural-simplification/tree/review/post-v02-independent-2026-09-12) | Draft [PR #100](https://github.com/Unjuno/canaria-neural-simplification/pull/100): separate internal claim review, not external peer review. |
| [`publication/announcement-candidate-2026-09-12`](https://github.com/Unjuno/canaria-neural-simplification/tree/publication/announcement-candidate-2026-09-12) | Draft [PR #101](https://github.com/Unjuno/canaria-neural-simplification/pull/101): separately scoped R87R2/Phase4 announcement proposal; not merged by the October integration. |

No research becomes a universally valid public claim merely because its files exist on main. Main integration makes the checkpoint inspectable; it does not merge cohorts, alter preregistration, or silently adopt every research PR. The September PR #101 still needs a new live-disposition review if resumed.

## Where information is authoritative

| Information | First stop |
|---|---|
| October Digits/CIFAR interpretation | [Checkpoint assessment](../RESEARCH_CHECKPOINT.md), [dated report](../research-checkpoints/2026-10-27/README.md) |
| What was and was not reproduced | [Status](../STATUS.md), [reproducibility boundaries](REPRODUCIBILITY.md) |
| Historical baseline and corrections | [Historical index](HISTORICAL_INDEX.md), [original discrepancy](REPRODUCTION_DISCREPANCY.md) |
| Cross-family and branch-pinned evidence | [Research index](RESEARCH_INDEX.md) |
| Experimental packages and dated receipts | [Research packages](../research/README.md), [checkpoints](../research-checkpoints/README.md) |
| Archived code/results and original paths | [Archive guide](../archives/README.md), [migration map](../publication/PATH_MIGRATION.json) |
| PR ownership and scope | [Workstreams](WORKSTREAMS.md), [integration record](INTEGRATION_2026-10-31.md) |

## Evidence lifecycle and safe cleanup

A protocol establishes planned conditions; raw seed records establish observations; a recalculation checks the stored decision. None is automatically external peer review. A later integration commit must never be presented as original preregistration.

- Preserve original scientific code, raw outputs, protocol locks, archives and correction markers. Do not rewrite FAIL/UNCERTAIN/INVALID as PASS.
- Keep the frozen tag, historical reproduction **Issue #87** ([tracker](https://github.com/Unjuno/canaria-neural-simplification/issues/87)) and announcement [Issue #13](https://github.com/Unjuno/canaria-neural-simplification/issues/13) separate from repository maintenance.
- Do not force-push, mass-delete branches, or mechanically merge unrelated research PRs. Completed source branches are retained for immutable-link and provenance consumers.
- Generated scratch outputs belong under ignored `outputs/`; tracked evidence retains its original path.
- Open-PR/branch counts in older documents are dated snapshots, not current authority to merge or delete anything.

## Maintenance checks

Run `python -m unittest discover -s tests -v`, `python tools/audit_repo.py`, and `python tools/audit_publication.py`. The latter checks historical/editorial integrity; its strict `--require-reproduction` path remains unresolved and is not redefined.

The added `checkpoint-integration-preservation` workflow verifies original Git blobs/modes from old main and the organized checkpoint, rejects deletions, checks ancestor history and the frozen tag, and checks retained historical policy boundaries. Exact-head CI and post-merge CI must be inspected separately; neither certifies scientific readiness or a release.
