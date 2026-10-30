# Canaria repository operations — checkpoint navigation

> Editorial and operational map, reviewed 2026-10-31 (JST). This is not a new scientific result, independent replication, merge approval, or release certificate.

## Choose the correct Git reference

| Git reference | Role |
|---|---|
| [`main`](https://github.com/Unjuno/canaria-neural-simplification/tree/main) | Public historical baseline and preserved correction/reproduction notices; do not mistake it for the October research report. |
| [`v0.2.0-public-snapshot`](https://github.com/Unjuno/canaria-neural-simplification/tree/v0.2.0-public-snapshot) | Immutable historical tag; do not retag. |
| [`research/checkpoint-20261026`](https://github.com/Unjuno/canaria-neural-simplification/tree/research/checkpoint-20261026) | October research collection, including Digits L/J/C, unsuccessful CIFAR-10 experiments and source/checkpoint records. |
| [`docs/checkpoint-handoff-20261030`](https://github.com/Unjuno/canaria-neural-simplification/tree/docs/checkpoint-handoff-20261030) | 14-document editorial handoff, Draft [PR #106](https://github.com/Unjuno/canaria-neural-simplification/pull/106), stacked on the checkpoint, not on `main`. |
| [`review/post-v02-independent-2026-09-12`](https://github.com/Unjuno/canaria-neural-simplification/tree/review/post-v02-independent-2026-09-12) | Draft [PR #100](https://github.com/Unjuno/canaria-neural-simplification/pull/100): internal claim review, not external peer review. |
| [`publication/announcement-candidate-2026-09-12`](https://github.com/Unjuno/canaria-neural-simplification/tree/publication/announcement-candidate-2026-09-12) | Draft [PR #101](https://github.com/Unjuno/canaria-neural-simplification/pull/101): separately scoped R87R2/Phase4 announcement candidate, not the October L/J/C report. |

**These branches are not interchangeable or automatically cumulative.** No research is promoted into the main claim registry merely because a branch, PR or CI check exists. The latest review of PR #101 listed 30 open PRs; the live count inspected for this map was 33, so its live-disposition gate needs a new scientific/editorial decision before use.

## Where information is authoritative

| Information | First stop |
|---|---|
| Current October Digits/CIFAR scientific interpretation | [Checkpoint assessment](../RESEARCH_CHECKPOINT.md), then the [full dated report](../research-checkpoints/2026-10-27/README.md) |
| What was and was not reproduced | [Checkpoint status](../STATUS.md), [reproducibility boundaries](REPRODUCIBILITY.md) |
| Historical baseline and corrections | [Historical index](HISTORICAL_INDEX.md), [original discrepancy](REPRODUCTION_DISCREPANCY.md) |
| Cross-family research and branch-pinned evidence | [Whole-project index](RESEARCH_INDEX.md) |
| Experimental source packages | [Research package guide](../research/README.md) |
| Dated report/reproduction receipts | [Checkpoint guide](../research-checkpoints/README.md) |
| Archived code/results and old paths | [Archive guide](../archives/README.md), [migration map](../publication/PATH_MIGRATION.json) |
| Open PR ownership and classification | [Workstreams/triage](WORKSTREAMS.md) |

## Evidence lifecycle

1. A protocol or source snapshot establishes the **planned** conditions; a later commit cannot be represented as the original preregistration.
2. Per-seed raw outputs and immutable hashes document **what was observed**; summaries are not substitutes for missing inputs.
3. An independent recalculation checks recorded decisions; it is not new scientific replication.
4. A review ledger explicitly classifies KEEP / EDIT / EXCLUDE / INVALIDATE before changing active claims.
5. A publication PR and its exact-head CI establish software integrity only. Main promotion, peer review, release and external announcement are separate decisions.

## Safe cleanup rules

- Preserve existing scientific code, output JSON/NPZ, raw evidence, original protocols, archived history and correction markers. Do not rewrite a failed or invalidated result to PASS.
- Keep `v0.2.0-public-snapshot`, historical reproduction Issue #87 ([tracker](https://github.com/Unjuno/canaria-neural-simplification/issues/87)) and announcement tracker [#13](https://github.com/Unjuno/canaria-neural-simplification/issues/13) distinct from newer confirmations.
- Treat the 90 branches and 33 open PRs observed on 2026-10-31 as a **dated snapshot**, not a permanent count or permission to delete them.
- Never force-push, mass-delete branches, mechanically merge research PRs, or change the default branch merely to shorten a directory listing.
- Close or archive a superseded PR only after its provenance, replacement reference, and open scientific obligations have been reviewed and recorded.
- Keep generated scratch outputs in the existing ignored `outputs/` area; tracked evidence stays in its original scientific path.

## Minimal maintenance checks

From a checkout of the target branch, use `python -m unittest discover -s tests -v`, `python tools/audit_repo.py`, and `python tools/audit_publication.py`. The last command audits baseline/editorial integrity; historical strict exact reproduction remains a separate unresolved gate (`--require-reproduction`) and must not be redefined.

This page does not authorize running expensive experiments, altering protected evidence, or publishing a release.
