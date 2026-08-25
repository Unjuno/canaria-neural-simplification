# Changelog

Canaria is a research repository rather than a conventional production library. Versions mark research snapshots and evidence organization, not API-stability guarantees.

## 0.2.0 — 2026-08-25 — public research snapshot

### Research framing

- Reframed the project around **task-conditioned compositional simplification of learned neural computation**.
- Separated the core compositional finding from the later training-time/dynamic extension.
- Clarified that the project does **not** claim universal mathematical/Kolmogorov complexity reduction under composition.

### Direct cross-family replication of the core discovery

- Added a fresh confirmatory Small Vision Transformer experiment directly comparing component-wise versus composed replacement of the same fixed central two-block span.
- Locked passing criterion before confirmatory outcomes: training-held-out span NMSE `<=0.12` and validation utility `>=0.95`.
- Fresh seed rule: first 8 baseline-eligible seeds `>=9000`; exploratory 8900-series excluded; test data not used for candidate selection.
- Result:
  - component-wise minimum passing complexity: **9,808 replacement params** in all 8 seeds;
  - composed minimum passing complexity: **4,904–5,424 params**;
  - mean composed/component-wise complexity ratio: **0.51988**;
  - paired seed-bootstrap95: **[0.50634, 0.53926]**;
  - composed smaller in **8/8** fresh seeds;
  - selected composed mean test utility: **0.97856**, bootstrap95 **[0.97090, 0.98562]**;
  - component-wise compiler updates **640** vs composed **320**.
- Primary pre-registered decision: **PASS**.
- Added:
  - `docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md`
  - `scripts/replication/vit_compositional.py`
  - `results/replication/vit_compositional/PROTOCOL_LOCK.json`
  - `results/replication/vit_compositional/confirm_summary.json`
  - `results/replication/vit_compositional/seed_table.csv`
- Closed GitHub Issue #2 as completed.

### Training-time consolidation evidence

Public documentation incorporates the G7–G27 program, including progressive training-time consolidation, function-aligned transfer controls, staged-vs-direct path effects, factorization-without-learning equivalence control, autonomous/deadline-aware controllers, recontracting-dependent compiler conditioning and task sensitivity, sensitivity-aware immediate task-risk prediction, horizon-aware future-damage prediction, and negative controller results (G21 and G27).

### Reproducibility closure

- Added a self-contained public runner for G7 fresh confirmatory seed 4300 with no private `/mnt/data` imports.
- Added a pinned reproduction environment and manual GitHub Actions workflow.
- The portable run reproduced the archived confirmatory JSON exactly in the recorded environment, with matching SHA256:
  `68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`.
- Closed GitHub Issue #1 as completed.

### Runtime/materialization proof of concept

- Added a bounded G7 seed-4300 CPU PoC that serializes, materializes, and directly executes the learned compact 2-block representation.
- Serialized artifact + manifest decreased **110,093 → 54,646 bytes** (`−50.36%`).
- Parameters decreased **23,138 → 11,042** (`−52.28%`).
- Batch-128 CPU inference decreased **47.05 → 23.11 ms mean** over five fresh-process probes.
- Process RSS delta changed only **4.72 → 4.56 MB**, so meaningful host-RAM reduction was **not demonstrated**.
- Closed GitHub Issue #3 as completed.

### Evidence organization

Added or substantially revised the public snapshot, core discovery, claim registry, publication notes, terminology/FAQ, training-time findings, negative results, applications, runtime PoC, reproducibility, roadmap, open questions, historical index, release checklist, late-stage manifests, and replication/reproduction artifacts.

### Public handoff and scientific closure

- Broad experiment expansion for v0.2.0 is stopped.
- All three bounded closure tasks are complete:
  1. representative clean-repository reproduction;
  2. direct cross-family replication of the core compositional-simplification phenomenon;
  3. bounded runtime/materialization PoC.
- No additional experiment is required to freeze the current scientific claim scope.
- Future work should start as a new issue/research phase rather than extending the old G-number mainline.

### Integrity

- Upgraded `tools/audit_repo.py` to require current public-snapshot documents, late-stage evidence, portable reproduction artifacts, runtime-PoC artifacts, and the fresh SmallViT replication artifacts.
- Added semantic checks preserving the G21 failure, reproduction/PoC interpretation boundaries, and the 8/8 fresh composed-lower replication result.
- Added public-Markdown link checks and guards against reintroducing private `/mnt/data` dependencies in portable runners.
- Repository CI runs the audit and reusable unit tests.

### Known limitations

- The exact G7 portability reproduction validates one already-confirmatory seed; it is not a new independent scientific replication.
- The SmallViT replication is still small-model, task-manifold, and replacement-grammar dependent; it does not establish universal Transformer or LLM subadditivity.
- The runtime PoC is small-model and CPU-only; it does not establish GPU, LLM, energy, peak-RAM, or universal runtime benefits.
- Some late-stage raw protocol/result artifacts are indexed by retained SHA256 rather than duplicated as separate repository files.

## 0.1.0 — 2026-08-24 — training-time mainline snapshot

- Published the corrected training-time consolidation mainline through G17.
- Added autonomous-controller evidence, staged-vs-direct controls, and factorization equivalence evidence.
- Preserved v23–v25 natural-text autoregressive failures as boundary results.
- Published Apache-2.0 repository structure, citation metadata, historical evidence, and reusable research components.
