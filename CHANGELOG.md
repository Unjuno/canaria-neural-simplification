# Changelog

Canaria is a research repository rather than a conventional production library. Versions mark research snapshots and evidence organization, not API-stability guarantees.

## 0.2.0 — 2026-08-25 — public research snapshot

### Research framing

- Reframed the project around **task-conditioned compositional simplification of learned neural computation**.
- Separated the core compositional finding from the later training-time/dynamic extension.
- Clarified that the project does **not** claim universal mathematical/Kolmogorov complexity reduction under composition.

### Direct cross-family replication of the core discovery

#### Small Vision Transformer

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

#### Residual MLP

- Added a second fresh confirmatory architecture-family replication using a four-block residual MLP on sklearn digits and a fixed first-two-block span.
- Component-wise and composed replacements had **exactly matched learned replacement-parameter counts** at every budget-grid point.
- Candidate selection used validation span NMSE `<=0.08` and validation accuracy within 2 absolute percentage points of the teacher; test accuracy was not used for budget selection.
- Fresh confirmatory seeds: `1200–1207`; exploratory seeds `1100–1103` excluded from inference.
- Result:
  - component-wise mean minimum passing budget: **3584 params**;
  - composed mean minimum passing budget: **1728 params**;
  - composed smaller in **8/8 fresh seeds**;
  - mean `log2(B_composed/B_componentwise)`: **−1.0519**;
  - paired seed-bootstrap95: **[−1.2075, −0.8962]**;
  - geometric mean budget ratio: **0.4823×**;
  - untouched-test accuracy difference at selected budgets: **+0.583 percentage points**, bootstrap95 **[+0.306,+0.806] pt**.
- Mechanistic secondary at fixed 2048 params:
  - local component-wise NMSE: **0.1474**;
  - same two-module architecture jointly fit to composed span target: **0.0639**;
  - one composed module: **0.0533**.
- This control strengthens the interpretation that much of the effect follows the **functional span objective/boundary**, not merely a one-module topology change.
- Added:
  - `docs/CORE_DISCOVERY_REPLICATION_DIGITS.md`
  - `scripts/reproduce/core_discovery_digits/run_confirmatory.py`
  - `results/core_discovery_digits/PROTOCOL_LOCK.json`
  - `results/core_discovery_digits/confirm_summary.json`

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
- All bounded closure tasks are complete.
- The original residual-CNN core discovery now has direct fresh architecture-family replications in both a Small Vision Transformer and a residual MLP.
- No additional experiment is required to freeze the current scientific claim scope.
- Future work should start as a new issue/research phase rather than extending the old G-number mainline.

### Integrity

- Upgraded `tools/audit_repo.py` to require current public-snapshot documents, late-stage evidence, portable reproduction artifacts, runtime-PoC artifacts, the SmallViT replication artifacts, and the residual-MLP direct-replication artifacts.
- Added semantic checks preserving the G21 failure, reproduction/PoC interpretation boundaries, and both 8/8 fresh composed-lower replication results.
- Added public-Markdown link checks and guards against reintroducing private `/mnt/data` dependencies in portable runners.
- Repository CI runs the audit and reusable unit tests.

### Known limitations

- The exact G7 portability reproduction validates one already-confirmatory seed; it is not a new independent scientific replication.
- The SmallViT and residual-MLP replications are still small-model, task-manifold, and replacement-grammar dependent; they do not establish universal Transformer, LLM, task-universal, or grammar-independent subadditivity.
- Both direct replications use sklearn-digits supervised classification; task-type external validity remains open.
- The runtime PoC is small-model and CPU-only; it does not establish GPU, LLM, energy, peak-RAM, or universal runtime benefits.
- Some late-stage raw protocol/result artifacts are indexed by retained SHA256 rather than duplicated as separate repository files.

## 0.1.0 — 2026-08-24 — training-time mainline snapshot

- Published the corrected training-time consolidation mainline through G17.
- Added autonomous-controller evidence, staged-vs-direct controls, and factorization equivalence evidence.
- Preserved v23–v25 natural-text autoregressive failures as boundary results.
- Published Apache-2.0 repository structure, citation metadata, historical evidence, and reusable research components.
