# Research and pull-request workstreams

> Open-PR triage snapshot: 2026-10-31 JST. It is a navigation and review queue, not a claim of scientific validity or an instruction to merge/close anything.

At inspection: **33 open PRs** and **90 branches**. A GitHub issue/PR being open does not mean the associated experiment is still running. Refer to each protocol/result for its scientific status. The live list can change at any time: [open PRs](https://github.com/Unjuno/canaria-neural-simplification/pulls), [branches](https://github.com/Unjuno/canaria-neural-simplification/branches).

| Workstream | Open PRs at snapshot | Review disposition / next action |
|---|---|---|
| Publication claim-review stack | [#100](https://github.com/Unjuno/canaria-neural-simplification/pull/100), [#101](https://github.com/Unjuno/canaria-neural-simplification/pull/101), [#106](https://github.com/Unjuno/canaria-neural-simplification/pull/106) | **Drafts, different bases**. Review in dependency order; #101 is not a merger of the October checkpoint and #106 is docs-only. |
| Older maintenance proposals | [#14](https://github.com/Unjuno/canaria-neural-simplification/pull/14), [#17](https://github.com/Unjuno/canaria-neural-simplification/pull/17) | Compare against merged [#88](https://github.com/Unjuno/canaria-neural-simplification/pull/88); mark superseded only after verifying remaining differences. |
| Reproduction / numerical environment | [#89](https://github.com/Unjuno/canaria-neural-simplification/pull/89), [#90](https://github.com/Unjuno/canaria-neural-simplification/pull/90) | Retain diagnosis/new numerical profile. Historical exact cohort still [Issue #87](https://github.com/Unjuno/canaria-neural-simplification/issues/87). |
| Regression / teacher validity | [#11](https://github.com/Unjuno/canaria-neural-simplification/pull/11), [#91](https://github.com/Unjuno/canaria-neural-simplification/pull/91), [#93](https://github.com/Unjuno/canaria-neural-simplification/pull/93), [#95](https://github.com/Unjuno/canaria-neural-simplification/pull/95) | Preserve strong and weak/negative teacher conditions separately. Phase4 is bounded dataset/task evidence. |
| SmallViT and rank mechanism | [#104](https://github.com/Unjuno/canaria-neural-simplification/pull/104), [#105](https://github.com/Unjuno/canaria-neural-simplification/pull/105) | Synthetic rank null PASS is limited to a linear oracle; fixed-topology SmallViT objective advantage was UNCERTAIN. Neither is a universal compression theorem. |
| Recursive / systems integration | [#56](https://github.com/Unjuno/canaria-neural-simplification/pull/56) | Review as supporting research and bounded prototypes, not general hardware claims. |
| Gaussian shift / correction interfaces | [#68–#86](https://github.com/Unjuno/canaria-neural-simplification/pulls) | 19 historical research PRs; #68 imported CNN lineage has a provenance gap, #69–#86 include distinct Residual-MLP exploratory/confirmatory conditions. Do not pool or mass-merge. |

**Total: 3 + 2 + 2 + 4 + 2 + 1 + 19 = 33 open PRs** as of inspection. Any future PR must be triaged separately before an announcement: a dated index cannot substitute for a live review.

## Branch prefixes

- `research/*`: study-specific source, protocols, results, and experimental branches; some branches may be standalone evidence not tied to an open PR.
- `execution/*`: bounded experimental execution scaffolding; do not remove until run provenance and dependency references are checked.
- `review/*`: internal re-evaluation and claim selection.
- `publication/*`: public-claim candidates and policy gates, not automatic release authority.
- `docs/*` and `maintenance/*`: navigation, evidence preservation and editorial cleanup.
- `integration/*`: reviewed collections or earlier integration proposals, not default-branch scientific promotion.

## Backlog and safe dispositions

1. **Now:** review the two newest mechanism PRs and the October checkpoint against the frozen public-claim registry. Keep their findings in separate evidence classes.
2. **Now:** update the [#101](https://github.com/Unjuno/canaria-neural-simplification/pull/101) open-PR snapshot *on its own branch* only after deciding how to treat PRs #104–#106; do not weaken its fail-closed live check.
3. **Then:** inspect maintenance PRs #14/#17 against merged #88 and identify any truly superseded proposals. Do not close them based on age alone.
4. **Later:** inventory orphan branches, workflows and source-only packages by their producing commits before proposing deletions; historical hashes and archived release must survive.

## Scientific boundaries

Current October L/J/C selected-budget results, R87R2's distinct 16-seed numerical-profile cohort, the historical 1200–1207 archive, SmallViT, California Housing and synthetic rank oracles use **different experimental units and comparison denominators**. Read [checkpoint](../RESEARCH_CHECKPOINT.md), [historical reproducibility notice](REPRODUCTION_DISCREPANCY.md), and [research index](RESEARCH_INDEX.md) before combining anything in a public summary.
