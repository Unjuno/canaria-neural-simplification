# Repository layout

Canaria is organized as an auditable research repository. Paths are kept stable where possible because protocols, reviews, issues, result hashes, and external citations may refer to them directly.

## Authoritative surfaces

- `README.md` — concise project statement, scope, and entry points.
- `STATUS.md` — current reviewed public-baseline state and scientific boundaries.
- `QUICKSTART.md` — shortest supported reproduction path.
- `docs/CLAIMS_AND_EVIDENCE.md` — authoritative public claim registry.
- `docs/README.md` — documentation index and current-versus-historical reading guide.

The frozen release tag `v0.2.0-public-snapshot` is immutable. Later work must not rewrite that tag or back-project later corrections into it.

## Directory roles

| Path | Role | Mutation policy |
| --- | --- | --- |
| `docs/` | Current interpretation, reviews, protocols, phase narratives, and preserved research history | Current index/interpretation files may be updated; locked protocols and historical records should be preserved |
| `results/` | Machine-readable evidence, protocol locks, summaries, correction records | Evidence-producing result files are append-only/immutable after use in inference; corrections are added explicitly |
| `scripts/reproduce/` | Clean-clone/public reproduction runners | May receive additive portability fixes or new versioned runners |
| `scripts/replication/` | Direct replication runners | Preserve evidence-producing versions; add new versions instead of silently rewriting old outcomes |
| `scripts/phase2/` | Phase 2 precision/quantization runners associated with the reviewed correction boundary | Preserve provenance; do not rehabilitate invalidated evidence by rewriting code in place |
| `scripts/phases/` | Phase-specific and historical experiment code | Primarily evidence/provenance code; environment assumptions may be historical |
| `src/canaria/` | Reusable, cleaned library code separated from historical experiments | Normal library-quality maintenance with tests |
| `tests/` | Tests for reusable code and stable repository behavior | Normal maintenance |
| `tools/` | Repository/evidence audits and verification utilities | Normal maintenance; audit semantics should stay explicit |
| `schemas/` | Machine-readable metadata/event schemas | Version or document incompatible changes |
| `environment/` | Environment/provenance records | Historical environment records are preserved |
| `archives/` | Hashes and archive provenance | Preserve identifiers; do not substitute archive material for checked evidence without documentation |
| `.github/` | Contribution governance and stable CI/workflow entry points | Stable CI may remain; one-shot research workflows should not accumulate after their evidence is committed |

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
public claim registry, only if reviewed and merged
```

Exploration and confirmation should be distinguishable in both filenames/metadata and interpretation. Failed, uncertain, or invalidated evidence remains in the record with an explicit status rather than being silently removed.

## Current versus historical material

Do not infer authority from a high version number or a recent-looking filename.

Use these rules:

1. `docs/CLAIMS_AND_EVIDENCE.md` and `STATUS.md` define the reviewed public surface.
2. `docs/HISTORICAL_INDEX.md` identifies planning/theory documents that are preserved for provenance rather than current instruction.
3. `results/README.md` explains which result families are current headline evidence, reproduction evidence, correction history, or historical extended phases.
4. Locked protocol/result artifacts remain authoritative for what a particular experiment actually did even if later interpretation changes.

## Branch and PR policy

- `main` is the reviewed public baseline.
- New scientific work belongs on an isolated research branch and usually a draft PR until its stopping rule and review are complete.
- Unmerged research results do **not** alter `main`, the public claim registry, or the frozen release boundary.
- Repository organization, documentation cleanup, CI maintenance, and similar science-neutral work should use a separate maintenance branch/PR rather than being mixed into fresh experimental commits.
- A research PR that passes its own protocol is not automatically merge-ready; scientific review and claim-boundary review remain separate steps.

## Workflow lifecycle

Stable workflows such as repository-wide CI or intentionally supported reproduction jobs may live on `main`.

One-shot workflows used to execute a particular exploration/confirmation are execution scaffolding, not long-term product surface. After their outputs and provenance are committed, remove those one-shot workflow files from the branch tip when they are no longer needed. Git history preserves the exact workflow used for the run.

## Path-stability rule

Prefer adding indexes and role documentation over mass-renaming historical directories. Rename or relocate evidence paths only when the benefit clearly outweighs broken references, and record a migration map when doing so.
