# Historical document index

Canaria intentionally preserves earlier roadmaps, theory drafts, transfer plans, and experiment ledgers. These files are useful for provenance, but they are **not all current project instructions**.

Use `README.md` in this directory, `PUBLIC_SNAPSHOT.md`, `CLAIMS_AND_EVIDENCE.md`, `ROADMAP.md`, and repository-root `STATUS.md` for the current public state.

## Current authoritative documents

- `README.md` — documentation-directory index.
- `PUBLIC_SNAPSHOT.md` — reading order and snapshot policy.
- `CORE_DISCOVERY.md` — central discovery and scope.
- `CLAIMS_AND_EVIDENCE.md` — current claim registry.
- `PUBLICATION_NOTES.md` — publication-safe claim hierarchy.
- `TRAINING_TIME_CONSOLIDATION.md` — G7–G17 evidence.
- `LATE_STAGE_FINDINGS.md` — G18–G27 evidence.
- `NEGATIVE_RESULTS.md` — current negative/boundary registry.
- `TERMINOLOGY.md` — current definitions.
- `FAQ.md` — interpretation boundaries.
- `APPLICATIONS.md` — application directions separated by evidence status.
- `RUNTIME_POC.md` — bounded small-model CPU systems PoC.
- `REPRODUCIBILITY.md` — current reproduction policy and portable G7 path.
- `ROADMAP.md` — current closure/handoff roadmap.
- `OPEN_QUESTIONS.md` — unresolved questions for future researchers.

## Completed closure artifacts

Two formerly open closure tasks are complete:

- Issue #1 — clean-repository reproduction of G7 fresh confirmatory seed 4300;
- Issue #3 — minimal runtime/materialization/direct-execution PoC.

Only Issue #2 remains optional: direct replication of compositional simplification on a clearly different architecture/task if a stronger publication-level generalization/novelty claim is desired.

## Preserved historical planning / status documents

### `GENERALIZATION_ROADMAP.md`

A prespecified cross-architecture transfer roadmap written before later G5/G6/v23–v25 and training-time consolidation results were known. Its adaptation taxonomy remains useful, but its listed future sequence is historical rather than a current commitment.

### `GENERALIZATION_STATUS.md`

The historical transfer ledger through the post-hoc Transformer/natural-text program. Its "next tests" section records the frontier at that time. Statements such as the old clean-room-reproduction status reflect that historical moment and are superseded by current `STATUS.md` / `REPRODUCIBILITY.md`.

### `NEXT_EXPERIMENTS_AUTONOMOUS.md`

A planning document from the G18/G19 frontier. Those experiments have since been run; current outcomes are in `LATE_STAGE_FINDINGS.md` and `results/training_time/late_stage_summary.json`.

### `RESEARCH_SUMMARY.md`

A broad historical summary. Useful for chronology and provenance; not the shortest route to the current claims.

### `KNOWLEDGE_MAP.md`

Earlier conceptual organization of mechanisms and open questions. Treat as historical theory context, not as the current claim registry.

### `DATA_DICTIONARY.md`

Definitions for historical result tables and fields. Still useful when reading archived phase outputs.

## Phase and history directories

- `docs/phases/` — locked phase protocols/results and detailed evidence.
- `docs/history/` — preserved handoff/theory documents from earlier snapshots.
- `archives/` — retained archival bundles and provenance material.

## Preservation rule

Do not silently rewrite historical protocols, outcome ledgers, or evidence-producing scripts to make them look consistent with later theory. When interpretation changes, add a current document that explains the change and keep the old artifact intact.
