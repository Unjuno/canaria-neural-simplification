# Changelog

Canaria is a research repository rather than a conventional production library. Versions mark research snapshots and documentation/evidence organization, not API-stability guarantees.

## 0.2.0 — 2026-08-25 — public research snapshot

### Research framing

- Reframed the project around the central empirical observation of **task-conditioned compositional simplification of learned neural computation**.
- Separated the core discovery from the later dynamic/training-time extension.
- Clarified that the project does **not** claim universal mathematical/Kolmogorov complexity reduction under composition.

### Training-time consolidation evidence

Public documentation now incorporates the G7–G27 program, including:

- progressive training-time consolidation;
- function-aligned transfer controls;
- staged-vs-direct path effects;
- factorization-without-learning equivalence control;
- autonomous/deadline-aware controllers;
- recontracting-dependent compiler conditioning and task sensitivity;
- sensitivity-aware immediate task-risk prediction;
- horizon-aware future-damage prediction;
- negative controller results (G21 and G27).

### Evidence organization

Added or substantially revised:

- `docs/PUBLIC_SNAPSHOT.md`
- `docs/CORE_DISCOVERY.md`
- `docs/CLAIMS_AND_EVIDENCE.md`
- `docs/PUBLICATION_NOTES.md`
- `docs/TERMINOLOGY.md`
- `docs/FAQ.md`
- `docs/LATE_STAGE_FINDINGS.md`
- `docs/NEGATIVE_RESULTS.md`
- `docs/APPLICATIONS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/ROADMAP.md`
- `docs/HISTORICAL_INDEX.md`
- `results/training_time/late_stage_summary.json`
- `results/training_time/ARTIFACT_INVENTORY.md`

### Reproducibility closure

- Added `scripts/reproduce/g7_confirmatory/run_seed.py`, a self-contained public runner for G7 fresh confirmatory seed 4300 with no private `/mnt/data` imports.
- Added `results/reproduction/g7_seed4300_report.json`.
- The portable run reproduced the archived confirmatory JSON exactly in the recorded environment, with matching SHA256 `68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`.
- Added `.github/workflows/reproduce-g7.yml` as a manual exact-reproduction workflow.
- Updated snapshot/status/roadmap documentation to mark representative clean-repository reproduction as complete.

### Public handoff

- Broad experiment expansion is paused.
- Remaining closure work is conditional: one direct external replication only if a stronger generalization/novelty claim is pursued, and one runtime-compilation proof-of-concept only if deployment claims are pursued.
- Created narrow GitHub Issues for those conditional tasks.
- Historical plans/results remain preserved rather than rewritten to match the later theory.

### Integrity

- Upgraded `tools/audit_repo.py` to require current public-snapshot documents, late-stage evidence, G7 portable reproduction artifacts, and preservation of the G21 failure.
- Repository CI runs the audit and reusable unit tests.
- Added research-integrity issue/PR templates.
- Package/citation metadata advanced to `0.2.0`.

### Known limitations

- The exact G7 portability reproduction validates one already-confirmatory seed; it is not a new independent scientific replication.
- Some late-stage raw protocol/result artifacts are indexed by retained SHA256 rather than duplicated as separate repository files; see `results/training_time/ARTIFACT_INVENTORY.md`.
- Runtime-compilation, memory, latency, and energy benefits remain engineering hypotheses until benchmarked directly.

## 0.1.0 — 2026-08-24 — training-time mainline snapshot

- Published the corrected training-time consolidation mainline through G17.
- Added autonomous-controller evidence, staged-vs-direct controls, and factorization equivalence evidence.
- Preserved v23–v25 natural-text autoregressive failures as boundary results.
- Published Apache-2.0 repository structure, citation metadata, historical evidence, and reusable research components.
