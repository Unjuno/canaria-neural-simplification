# Verified research checkpoint — 2026-10-26

This branch publishes a **summary and independent-audit subset**, not the full local reproduction bundle. See [the file inventory](INVENTORY.json) for exact local source commits, Git blob IDs and SHA-256 hashes. Original preserved reports contain historical statements that publication was pending; this checkpoint supersedes those statements only for the files listed in the inventory. Raw training traces, captured tensors/artifacts, source snapshots and other referenced local evidence are **not all uploaded in this checkpoint**. Relative links in preserved reports may therefore remain unavailable here. Their audit claims describe local verification, not fresh verification using this reduced remote bundle.

## Results and limits

* R87R2: known bridge plus fixed 16-seed technical training reproduction; complete scientific outcomes and decision matched archived primary-host values. Replacement-budget geometric ratio 0.53161. This is not a fresh independent cohort and does not close historical Issue #87.
* Phase4: known preflight plus fixed eight-seed training reproduction; complete scientific records and decision matched archived evidence. Independent audit 154/154. Replacement-budget ratio 0.47603. The original full Phase4 source was needed; PR101 alone lacked its base runner. The separate packaging repair is not applied by this checkpoint.
* SmallViT objective-control remains UNCERTAIN: test noninferiority passes, NMSE improvement is not established. Saved-output audits do not prove historical test-access isolation.
* Native whole-model costs: pilot and three chronological known-seed extension measurements all fail the predeclared joint 5% artifact-size/5% median batch128-latency benefit requirement. Whole-artifact size ratios are approximately 0.96105, 0.96105, 0.96105, 0.95032; the last is not rounded into a pass. Only 2/16 frozen-cohort payload estimates meet 5% reduction. Payload is not archive size or latency; technical repeats are not new scientific N.

No universal halving law, whole-model halving, general RAM/energy/GPU improvement, independent external replication or release readiness is claimed. These records are research checkpoints, not a release or main-branch merge.

## Publication controls

Based on main c7fd4fb701064c9aeef5cc8109231d884e5258aa. All four workflow files were inspected before publication: repository-audit has push/pull_request triggers; the other three are workflow_dispatch-only. This commit uses [skip ci] and the new research branch is created directly at that commit. No PR or workflow dispatch is requested, and workflow settings/files remain unchanged. Post-write branch/hash and Actions readback are recorded separately by the publisher.
