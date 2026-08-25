# Changelog

Canaria is a research repository rather than a conventional production library. Versions mark research snapshots and evidence organization, not API-stability guarantees.

## 0.2.0 — 2026-08-25 — public research snapshot

### Research framing

- Reframed the project around **task-conditioned compositional simplification of learned neural computation**.
- Separated the core compositional finding from the later training-time/dynamic extension.
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

### Reproducibility closure

- Added `scripts/reproduce/g7_confirmatory/run_seed.py`, a self-contained public runner for G7 fresh confirmatory seed 4300 with no private `/mnt/data` imports.
- Added a pinned reproduction environment under `scripts/reproduce/g7_confirmatory/requirements.txt`.
- Added `results/reproduction/g7_seed4300_report.json`.
- The portable run reproduced the archived confirmatory JSON exactly in the recorded environment, with matching SHA256:
  `68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`.
- Added `.github/workflows/reproduce-g7.yml` as a manual exact-reproduction workflow.
- Closed GitHub Issue #1.

### Runtime/materialization proof of concept

- Added `scripts/reproduce/g7_confirmatory/runtime_poc.py`.
- Added `docs/RUNTIME_POC.md` and `results/reproduction/runtime_poc_seed4300_report.json`.
- Added `.github/workflows/runtime-poc.yml`.
- In the bounded G7 seed-4300 CPU PoC:
  - serialized artifact + manifest decreased **110,093 → 54,646 bytes** (`−50.36%`);
  - parameters decreased **23,138 → 11,042** (`−52.28%`);
  - batch-128 CPU inference decreased **47.05 → 23.11 ms mean** over five fresh-process probes;
  - load/materialize decreased **7.85 → 5.86 ms mean**, but cache sensitivity makes this secondary evidence;
  - process RSS delta changed only **4.72 → 4.56 MB**, so meaningful host-RAM reduction was **not demonstrated**;
  - the compact 2-block learned representation executes directly and does not reconstruct the original 4-block model.
- Closed GitHub Issue #3.

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
- `docs/RUNTIME_POC.md`
- `docs/REPRODUCIBILITY.md`
- `docs/ROADMAP.md`
- `docs/OPEN_QUESTIONS.md`
- `docs/HISTORICAL_INDEX.md`
- `results/training_time/late_stage_summary.json`
- `results/training_time/ARTIFACT_INVENTORY.md`

### Public handoff

- Broad experiment expansion is paused.
- Representative clean-repository reproduction and the bounded runtime PoC are complete.
- The only remaining optional closure item is GitHub Issue #2: direct replication of the **core compositional-simplification phenomenon** on a clearly different architecture/task, only if a stronger publication-level generalization/novelty claim is pursued.
- Historical plans/results remain preserved rather than rewritten to fit the later theory.

### Integrity

- Upgraded `tools/audit_repo.py` to require current public-snapshot documents, late-stage evidence, portable reproduction artifacts, and runtime-PoC artifacts.
- Added semantic checks preserving the G21 failure and reproduction/PoC interpretation boundaries.
- Repository CI runs the audit and reusable unit tests.
- Added research-integrity issue/PR templates.
- Package/citation metadata advanced to `0.2.0`.

### Known limitations

- The exact G7 portability reproduction validates one already-confirmatory seed; it is not a new independent scientific replication.
- The runtime PoC is small-model and CPU-only; it does not establish GPU, LLM, energy, peak-RAM, or universal runtime benefits.
- Some late-stage raw protocol/result artifacts are indexed by retained SHA256 rather than duplicated as separate repository files; see `results/training_time/ARTIFACT_INVENTORY.md`.

## 0.1.0 — 2026-08-24 — training-time mainline snapshot

- Published the corrected training-time consolidation mainline through G17.
- Added autonomous-controller evidence, staged-vs-direct controls, and factorization equivalence evidence.
- Preserved v23–v25 natural-text autoregressive failures as boundary results.
- Published Apache-2.0 repository structure, citation metadata, historical evidence, and reusable research components.
