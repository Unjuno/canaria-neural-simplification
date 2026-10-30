# Research and pull-request workstreams

> Maintenance record, 2026-10-31 JST. This is a navigation/review queue, not a claim of scientific validity or permission to merge all research.

## Completed integration path

- [PR #106](https://github.com/Unjuno/canaria-neural-simplification/pull/106): 14-document handoff merged into `research/checkpoint-20261026` at `d389f3209bbc58c02f79e687ef8313f96d82a993`.
- [PR #107](https://github.com/Unjuno/canaria-neural-simplification/pull/107): retargeted onto that checkpoint and merged at `5752f45d5e5896349f1c84bf8fefd8735fe70520`.
- [PR #108](https://github.com/Unjuno/canaria-neural-simplification/pull/108): owner-authorized main integration with exact-object preservation and final-head checks. Its GitHub state is the completion receipt.

The pre-integration 33-open-PR snapshot included #106 but preceded #107/#108. It is historical, not a live gate. The table below excludes these three integration PRs and lists the 32 separate proposals retained for review; check [live PRs](https://github.com/Unjuno/canaria-neural-simplification/pulls) for subsequent changes.

| Workstream | Retained PRs | Scope / next action |
|---|---|---|
| Separate publication review | [#100](https://github.com/Unjuno/canaria-neural-simplification/pull/100), [#101](https://github.com/Unjuno/canaria-neural-simplification/pull/101) | Distinct internal claim-selection and R87R2/Phase4 publication proposals. Not merged by #108; refresh their live review before resuming. |
| Older maintenance proposals | [#14](https://github.com/Unjuno/canaria-neural-simplification/pull/14), [#17](https://github.com/Unjuno/canaria-neural-simplification/pull/17) | Compare remaining differences against merged #88 and #108 before declaring them superseded. |
| Numerical reproduction | [#89](https://github.com/Unjuno/canaria-neural-simplification/pull/89), [#90](https://github.com/Unjuno/canaria-neural-simplification/pull/90) | Retain diagnostic/new-profile evidence. Historical strict reproduction remains Issue #87. |
| Regression / teacher validity | [#11](https://github.com/Unjuno/canaria-neural-simplification/pull/11), [#91](https://github.com/Unjuno/canaria-neural-simplification/pull/91), [#93](https://github.com/Unjuno/canaria-neural-simplification/pull/93), [#95](https://github.com/Unjuno/canaria-neural-simplification/pull/95) | Keep positive, uncertain and negative teacher regimes separate. |
| SmallViT / rank controls | [#104](https://github.com/Unjuno/canaria-neural-simplification/pull/104), [#105](https://github.com/Unjuno/canaria-neural-simplification/pull/105) | Linear-oracle result is not a universal neural compression law; fixed-topology objective improvement was UNCERTAIN. |
| Recursive / systems integration | [#56](https://github.com/Unjuno/canaria-neural-simplification/pull/56) | Supporting research and bounded prototypes, not general hardware savings. |
| Gaussian shift / correction interfaces | #68–#86: [open PR list](https://github.com/Unjuno/canaria-neural-simplification/pulls) | 19 research PRs. Imported CNN provenance and later MLP cohorts are distinct; no mass merge. |

## Branch prefixes and cleanup

`research/*` retains original study code/protocols/results; `execution/*` retains bounded execution scaffolding; `review/*` and `publication/*` retain distinct scientific/editorial proposals; `docs/*`, `maintenance/*` and `integration/*` record the organized integration path. A source branch is not disposable merely because its PR has merged: first verify immutable references, workflow dependencies and the merge ancestry.

The owner authorized integrating the October checkpoint, not deleting unrelated evidence or accepting all open scientific claims. Completed PRs are marked merged; historical and unrelated research branches are retained. Any future cleanup must record what supersedes the old material and preserve its exact source commit.

## Scientific boundaries

October L/J/C, R87R2, the old 1200–1207 cohort, SmallViT, California Housing and synthetic rank oracles have different inferential units and comparators. Do not pool them. See [checkpoint](../RESEARCH_CHECKPOINT.md), [historical reproducibility notice](REPRODUCTION_DISCREPANCY.md), and [research index](RESEARCH_INDEX.md).
