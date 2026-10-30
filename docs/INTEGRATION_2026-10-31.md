# October checkpoint integration — 2026-10-31 JST

## Authorization and scope

The repository owner explicitly requested completing integration and repository organization. The integration path is [PR #108](https://github.com/Unjuno/canaria-neural-simplification/pull/108); its merge status and exact-head Actions are the live completion evidence. Before that PR merges, this file describes the staged integration candidate.

The October research records and organized navigation become inspectable on main. This is **not** a new experiment, a blanket scientific approval, independent peer review, historical numerical recovery, a release tag, or an external announcement. The original negative/uncertain/invalid evidence stays intact.

## Preserved source history

| Reference | Commit |
|---|---|
| Main before this integration | `c7fd4fb701064c9aeef5cc8109231d884e5258aa` |
| October research checkpoint before documentation merge | `a56510dae611d79730d28b84a51f454027255f8e` |
| Documentation source, PR #106 | `7e2ec252b6fc3cb7e858adde5459e610f2243ca4` |
| Navigation source, PR #107 | `0c88988869364d7ae24f812309fd6b7e63853ca4` |
| PR #106 normal merge to checkpoint | `d389f3209bbc58c02f79e687ef8313f96d82a993` |
| PR #107 normal merge to checkpoint | `5752f45d5e5896349f1c84bf8fefd8735fe70520` |
| Frozen v0.2.0 historical tag | `556dce21c7a5516a16780cb28d528d1ff3968e53` |

The normal merge commits preserve original branch ancestry; no force-push, squash of scientific chronology, bulk branch deletion or outcome rewriting is used.

## Required technical verification

- Existing unit tests and navigation regressions.
- Existing structural and publication-preservation audits.
- Existing seed1200 smoke, explicitly **not** full-cohort reproduction.
- New [checkpoint-integration-preservation workflow](../.github/workflows/checkpoint-integration-preservation.yml): compares exact Git blob/mode identities against former main and organized checkpoint, permits only explicitly listed editorial changes, rejects deletion of any preexisting file, checks ancestor history and frozen tag, and checks historical Issue #87/no-peer-review/no-announcement policy fields.
- Inspection of the merged main's own CI, separate from pre-merge PR CI.

The preservation check includes mutation probes rejecting changed primary evidence and deleted files. It does not remeasure scientific performance or independently validate every historical claim. Gate reports are retained as Actions artifacts; exact workflow IDs and final main SHA are recorded on PR #108 after execution, avoiding a self-referential commit hash in this document.

## What remains separate

Issue #87 historical exact reproduction, Issue #13 broader announcement readiness, missing full-pipeline real-data reproduction inputs, and current CIFAR quality failures remain unresolved as documented. PRs #100/#101 and other research proposals are not automatically merged. Their decisions require separate review if pursued.

[Current status](../STATUS.md) · [Research assessment](../RESEARCH_CHECKPOINT.md) · [Repository map](../REPOSITORY_LAYOUT.md) · [Remaining workstreams](WORKSTREAMS.md)
