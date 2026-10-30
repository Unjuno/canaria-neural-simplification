# Research source and experiment packages

> These are retained research inputs, code, diagnostics and records. Directory presence is **not** a promise that an end-to-end experiment can run from a clean checkout.

Start with [current scientific assessment](../RESEARCH_CHECKPOINT.md), [full checkpoint report](../research-checkpoints/2026-10-27/README.md), [reproducibility limitations](../docs/REPRODUCIBILITY.md), and [whole-project evidence index](../docs/RESEARCH_INDEX.md).

| Family / package | Inspection route |
|---|---|
| Current report and bundled provenance | [`public_checkpoint_20261027/`](public_checkpoint_20261027/) |
| Digits controlled L/J/C training | [`digits_matched_source_20261027/`](digits_matched_source_20261027/) and [`digits_joint_budget_source_20261027/`](digits_joint_budget_source_20261027/) |
| Saved-scalar and threshold-only recomputation | [`public_saved_analysis_20261027/`](public_saved_analysis_20261027/), [`threshold_robustness_20261027/`](threshold_robustness_20261027/) |
| Whole-model storage and selected CPU timing source | [`selected_whole_storage_source_20261027/`](selected_whole_storage_source_20261027/), [`selected_latency_source_20261027/`](selected_latency_source_20261027/) |
| CIFAR teacher, preflight, held-out and replacement attempts | [`cifar10_teacher/`](cifar10_teacher/), [`cifar10_preflight/`](cifar10_preflight/), [`cifar10_heldout/`](cifar10_heldout/), [`cifar10_replacements/`](cifar10_replacements/), [`cifar10_utility_gap/`](cifar10_utility_gap/) |

Evidence may include source-only packages, cached numerical records, diagnostic data or partially completed attempts. **Do not infer executable completeness, fresh-cohort confirmation, test-set novelty or quality-gate PASS from the folder name.** In particular the recorded CIFAR-10 target remained unmet at its specified 2-percentage-point loss allowance.

Do not move or deduplicate these directories until exact upstream commit, data hashes, original relative links and references from published reports are accounted for. [Preservation policy](../docs/REPOSITORY_OPERATIONS.md).
