# Project status — integrated October research checkpoint

**Scientific report 2026-10-29; editorial organization 2026-10-30; main-integration change 2026-10-31 · limited research report, not a production release.**

[Current assessment and direction](RESEARCH_CHECKPOINT.md) · [Full methods/results report](research-checkpoints/2026-10-27/README.md) · [Integration record](docs/INTEGRATION_2026-10-31.md)

**研究一区切りの判断（2026-10-31）：** [限定的な結論・保存スカラー監査・停止条件・再開トリガー](docs/RESEARCH_MILESTONE_2026-10-31.md)。これは新しい科学的確認、旧再現債務の解消、リリース・告知ではありません。

The October checkpoint and documentation/navigation changes are integrated through [PR #108](https://github.com/Unjuno/canaria-neural-simplification/pull/108), following merged PRs #106 and #107. The PR state records whether a checkout is a pre-merge candidate or the integrated main. Repository integration is not scientific peer review, new experimental confirmation, release tagging, or an external announcement.

## What is ready to read

The current report describes the bounded 16-teacher Digits J/C comparison, substantial teacher-relative whole-model reduction but small additional C-over-J savings, conditional CPU timing, and the unsuccessful CIFAR-10 series. It is sufficient to share as a **limited research report with explicit limitations**, not as general compression success or completed independent replication.

This replaces the old entry-page emphasis on the eight-seed baseline; it does not merge the two cohorts or repair the baseline's unresolved numbers. [Historical entry pages](archives/README.md) retain the earlier wording.

## What has and has not been validated

| Item | Recorded status |
|---|---|
| Published scalar summaries and post-hoc threshold analysis | Recalculation code, inputs, and test records are available; not new scientific samples |
| Matched / grid / storage / latency source-only packages | Published with synthetic-test records and input requirements; not a complete original artifact bundle |
| Selected original artifacts | Limited existence/hash, tensor, and saved-logit consistency checks reported; not regenerated predictions from the weights |
| Fresh real-data end-to-end reproduction from public source | **Not completed.** The 2026-10-29 attempt ended during checkout preparation, before scientific stages began |
| External writer/generalization check | **Not completed:** acquisition did not succeed; no model result |
| Historical eight-seed baseline exact numerical reproduction | **Unresolved (Issue 87):** directional pattern persisted in two repeats, but some archived endpoints/statistics disagreed |
| Production utility / general compression | **Not established** |

The setup failure is an execution-preparation result, not a measured failure of Canaria's approximation method. Its cause and untested stages remain in the [supplemental reproduction record](research-checkpoints/2026-10-29/REPRODUCTION_STATUS.md). It does not change the recorded scientific estimates.

[Current report's validation boundary](research-checkpoints/2026-10-27/README.md) · [Historical reproduction discrepancy](docs/REPRODUCTION_DISCREPANCY.md) · [Historical baseline reports](results/reproduction/README.md)

## Historical gates and preservation

The repository is public; visibility, limited-report sharing, exact numerical reproduction, and a formal release are separate states. The [older announcement checklist](docs/ANNOUNCEMENT_READINESS.md) and Issue #13 concern the baseline preparation and are retained as historical gates, not evidence that the current limited report is scientifically complete. Main integration under the owner's explicit request does **not** close #13 or #87, authorize a new release, or post an announcement.

- Frozen tag `v0.2.0-public-snapshot` (`556dce21c7a5516a16780cb28d528d1ff3968e53`) is unchanged.
- Phase 2E stays **`INVALIDATED_IMPLEMENTATION_BUG` / `DO_NOT_USE_FOR_INFERENCE`**; Phase 2I causal attribution stays retracted; Phase 2O remains uncertain.
- Older correction-subspace, recursive-composition, systems, and regression studies remain separate in the [historical research index](docs/RESEARCH_INDEX.md). A smaller correction subspace is not a smaller whole model.
- Source protocols, code, numerical outcomes, missing results, and correction history are not rewritten by entry-page cleanup.

## Direction

Broader claims require separate evidence: a properly scoped full-pipeline reproduction, an unexposed independent confirmation path, or quality and whole-model resource gains on a defined target. CIFAR's examined interventions remain negative results, not instructions for endless retuning on an already-observed test set. See the [decision and explicit unknowns](RESEARCH_CHECKPOINT.md).

## Scope of this status page

This checkpoint status is not a verdict on every historical or branch-only family. See the [whole-project research index](docs/RESEARCH_INDEX.md). Whole-model parameter reduction relative to the teacher is about 44.14% for C and 43.43% for J; the additional C-over-J reduction is 1.26%. These are different comparisons.

## Later R87 evidence versus historical recovery

The [portable recipe / R87R2 result](https://github.com/Unjuno/canaria-neural-simplification/blob/56a8d1a50f15cad0d930cc899336071fa8efe7dc/results/reproduction/r87_resolution/RESULT.md) reports tested Intel/AMD numerical agreement and a distinct fresh 16-model-seed confirmation. It does not recover the original eight-seed archive, close its provenance discrepancy, or provide a new independent dataset. New scientific confirmation and historical exact reproduction are separate statuses.
