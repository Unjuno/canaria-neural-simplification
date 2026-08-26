# Repository layout

Canaria is maintained as an auditable research repository. The current surface is intentionally separated from the older experiment sequence so that a new reader does not have to infer authority from version numbers.

## Current authoritative surface

- `README.md` — project statement, strongest evidence, and current pre-announcement status.
- `STATUS.md` — current research/readiness state.
- `docs/ANNOUNCEMENT_READINESS.md` — active announcement gate.
- `docs/CLAIMS_AND_EVIDENCE.md` — reviewed baseline claim registry.
- `docs/README.md` — current documentation index.
- `results/README.md` — current machine-readable evidence index.

The tag `v0.2.0-public-snapshot` is immutable historical provenance. It is not a current announcement-readiness certificate.

## Active directories

| Path | Role |
| --- | --- |
| `docs/` | current interpretation, evidence summaries, corrections, readiness, and reference material |
| `results/core_discovery_digits/` | residual-MLP direct matched-budget evidence |
| `results/replication/` | direct cross-family replication evidence |
| `results/training_time/` | reviewed training-time evidence |
| `results/phase2/` | reviewed precision/quantization evidence and correction history |
| `results/reproduction/` | reproduction and bounded systems reports |
| `scripts/reproduce/` | supported clean-clone reproduction runners |
| `scripts/replication/` | direct replication runners |
| `scripts/phase2/` | Phase 2 runners that remain part of the reviewed evidence chain |
| `scripts/phases/training_time/` | training-time runners still referenced by current reviewed evidence |
| `src/canaria/` | reusable cleaned library code |
| `tests/` | reusable-code tests |
| `tools/` | evidence-integrity and readiness audits |
| `archives/` | historical release/review records and superseded/versioned research material |

## Archive boundary

A one-time pre-announcement restructure tracked by Issue #16 moved the older version-number research sequence out of the active `docs/`, `results/`, `scripts/`, and `environment/` navigation into `archives/research-history/`.

The move is organizational only. Historical blobs/results are retained; they are not rewritten to match later interpretation. `archives/README.md` contains the migration map.

Archived material may still contain old relative paths, old status language, historical environment assumptions, or hypotheses later rejected. Treat it as provenance, not current instruction.

## Evidence lifecycle

A confirmatory unit should have a visible chain:

```text
issue/question
  -> protocol lock
  -> evidence-producing runner
  -> immutable result artifacts
  -> audit/review
  -> bounded interpretation
  -> claim registry only after review
```

A reproduction run is separate. Re-running an already observed confirmatory seed under a pinned environment is portability/reproduction evidence; it does not silently increase the scientific confirmatory sample size.

Failed, uncertain, negative, and invalidated evidence is retained with status rather than deleted.

## Branch policy

- `main` is the reviewed evidence baseline, not an announcement certificate.
- New scientific work stays on an isolated research branch/draft PR until protocol, stopping rule, outcomes, and interpretation have been reviewed.
- Maintenance/repository cleanup stays separate from fresh scientific outcomes.
- A protocol PASS does not automatically promote an experiment into the headline claim set.
- Broad announcement requires the separate readiness gate.

## Workflow policy

Stable CI and intentionally supported reproduction workflows may remain active. One-shot execution workflows should be removed after their output/provenance has been retained. Git history preserves the exact workflow used.

## Path-migration rule

After the Issue #16 restructure, current evidence paths should remain stable unless there is a concrete auditability benefit to another move. Historical material belongs under `archives/`; do not move it back into active navigation merely because it contains a positive result.
