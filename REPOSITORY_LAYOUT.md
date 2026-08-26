# Repository layout

Canaria is organized as an auditable research repository. Paths are kept stable where possible because protocols, reviews, issues, result hashes, and external citations may refer to them directly.

## Authoritative surfaces

- `README.md` — concise project statement, scope, and current pre-announcement state.
- `STATUS.md` — current research/readiness state and scientific boundaries.
- `docs/ANNOUNCEMENT_READINESS.md` — active broad-announcement gate.
- `QUICKSTART.md` — shortest supported direct experiment path.
- `docs/CLAIMS_AND_EVIDENCE.md` — reviewed claim registry for the current baseline.
- `docs/README.md` — documentation index and current-versus-historical reading guide.

The frozen tag `v0.2.0-public-snapshot` is immutable. It is a historical research snapshot, not a current announcement-readiness certificate.

## Directory roles

| Path | Role | Mutation policy |
| --- | --- | --- |
| `docs/` | Current interpretation, reviews, protocols, phase narratives, readiness gates, and preserved research history | Current index/interpretation/readiness files may be updated; locked protocols and historical records should be preserved |
| `results/` | Machine-readable evidence, protocol locks, summaries, reproduction reports, correction records | Evidence-producing result files are append-only/immutable after use in inference; corrections are added explicitly |
| `scripts/reproduce/` | Clean-clone/pinned-environment reproduction runners | May receive additive portability fixes or new versioned runners; do not silently alter historical evidence-producing behavior |
| `scripts/replication/` | Direct replication runners | Preserve evidence-producing versions; add new versions instead of silently rewriting old outcomes |
| `scripts/phase2/` | Phase 2 precision/quantization runners associated with the reviewed correction boundary | Preserve provenance; do not rehabilitate invalidated evidence by rewriting code in place |
| `scripts/phases/` | Phase-specific and historical experiment code | Primarily evidence/provenance code; environment assumptions may be historical |
| `src/canaria/` | Reusable, cleaned library code separated from historical experiments | Normal library-quality maintenance with tests |
| `tests/` | Tests for reusable code and stable repository behavior | Normal maintenance |
| `tools/` | Repository/evidence/readiness audits and verification utilities | Normal maintenance; audit semantics should stay explicit |
| `schemas/` | Machine-readable metadata/event schemas | Version or document incompatible changes |
| `environment/` | Historical and current reproduction-environment provenance | Historical records are preserved; current reproduction records may be added |
| `archives/` | Hashes and archive provenance | Preserve identifiers; do not substitute archive material for checked evidence without documentation |
| `.github/` | Contribution governance and stable CI/workflow entry points | Stable CI may remain; one-shot research/maintenance workflows should not accumulate after their purpose is complete |

## Evidence lifecycle

A normal confirmatory research unit should have an explicit chain:

```text
question / issue
    ↓
protocol lock
    ↓
evidence-producing runner
    ↓
immutable per-seed / summary results
    ↓
audit / review
    ↓
current interpretation
    ↓
claim registry, only if reviewed and merged
```

A separate **reproduction lifecycle** may later rerun already-observed evidence under a pinned environment. Reproduction results must be labeled as reproduction and must not silently increase the confirmatory scientific seed count.

Exploration and confirmation should be distinguishable in both filenames/metadata and interpretation. Failed, uncertain, or invalidated evidence remains in the record with an explicit status rather than being silently removed.

## Current versus historical material

Do not infer authority from a high version number, a release name, or a recent-looking filename.

Use these rules:

1. `STATUS.md` and `docs/ANNOUNCEMENT_READINESS.md` define current readiness.
2. `docs/CLAIMS_AND_EVIDENCE.md` defines the reviewed claim baseline until a later scientific review changes it.
3. `docs/HISTORICAL_INDEX.md` identifies planning/theory documents preserved for provenance rather than current instruction.
4. `results/README.md` explains which result families are headline evidence, reproduction evidence, correction history, or historical extended phases.
5. Locked protocol/result artifacts remain authoritative for what a particular experiment actually did even if later interpretation changes.

## Branch and PR policy

- `main` is the reviewed **evidence baseline**, not by itself an announcement certificate.
- New scientific work belongs on an isolated research branch and usually a draft PR until its stopping rule and review are complete.
- Unmerged research results do **not** alter the claim registry or historical snapshot boundary.
- Repository organization, documentation cleanup, CI maintenance, and similar science-neutral work should use a separate maintenance branch/PR rather than being mixed into fresh experimental commits.
- A research PR that passes its own protocol is not automatically merge-ready; scientific review and claim-boundary review remain separate steps.
- Broad announcement requires the separate gate in `docs/ANNOUNCEMENT_READINESS.md`.

## Workflow lifecycle

Stable workflows such as repository-wide CI or intentionally supported reproduction jobs may live on `main`.

One-shot workflows used to execute a particular exploration, confirmation, reproduction audit, release-metadata change, or maintenance action are execution scaffolding. After their outputs and provenance are retained, remove those workflow files from the active branch tip when they are no longer needed. Git history preserves the exact workflow used for the run.

## Path-stability rule

Prefer adding indexes and role documentation over mass-renaming historical directories. Rename or relocate evidence paths only when the benefit clearly outweighs broken references, and record a migration map when doing so.