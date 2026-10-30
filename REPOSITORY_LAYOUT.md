# Repository layout

This map describes the October research collection integrated into `main` through [PR #108](https://github.com/Unjuno/canaria-neural-simplification/pull/108). Its source checkpoint is `research/checkpoint-20261026`; integration is not a new scientific release. The underlying code and evidence retain their original paths.

## Reading routes

| Goal | Entry point |
|---|---|
| Understand the project and its bounded results | [README](README.md) / [日本語](README.ja.md) |
| Read the scientific assessment and open questions | [Research assessment](RESEARCH_CHECKPOINT.md) |
| Find reports and separate historical studies | [Research index](docs/RESEARCH_INDEX.md) |
| Check what has actually been validated | [Status](STATUS.md) |
| Recompute saved scalar summaries without training | [Saved-result analysis](research/public_saved_analysis_20261027/README.md) |
| Inspect the separate historical eight-seed baseline | [Quickstart](QUICKSTART.md), [claim registry](docs/CLAIMS_AND_EVIDENCE.md), [reproduction discrepancy](docs/REPRODUCTION_DISCREPANCY.md) |
| Inspect this main integration and preserved history | [Integration record](docs/INTEGRATION_2026-10-31.md) |

## Code, reports, and evidence

| Path | Role |
|---|---|
| [`src/canaria/`](src/canaria/), [`tests/`](tests/) | Reusable code and tests |
| [`scripts/`](scripts/) | Scientific runners, including historical baseline and supporting studies; read the associated protocol before use |
| [`research-checkpoints/`](research-checkpoints/) | Dated reader-facing reports and supplemental records |
| [`research/`](research/) | Research source packages, protocols, saved results, and scalar-recomputation tools; some packages require inputs not included in the repository |
| [`results/`](results/) | Original baseline, supporting studies, reproduction records, and active correction/invalidation evidence |
| [`docs/`](docs/) | Interpretations, limitations, corrections, and navigation |
| [`environment/`](environment/), [`schemas/`](schemas/) | Environment records and data schemas |
| [`tools/`](tools/), [`publication/`](publication/) | Audits, scoped policies, catalogs and preservation/migration manifests |
| [`local_evidence/`](local_evidence/) | Existing tracked native-cost evidence; the directory name does not make its contents private |
| [`archives/`](archives/) | Preserved historical protocols, code, outcomes, and former entry/review/release documents |

## Maintainability and workstreams

[Repository operations](docs/REPOSITORY_OPERATIONS.md) identifies the purpose of each Git reference. [Workstream index](docs/WORKSTREAMS.md) distinguishes merged organization PRs from still-open research/publication proposals. [Research package guide](research/README.md) and [dated checkpoints](research-checkpoints/README.md) distinguish source-only packages from completed reports and unfinished reproduction attempts.

## Preservation and boundaries

The [archive guide](archives/README.md) and [migration map](publication/PATH_MIGRATION.json) retain older locations and context. This integration does not move, delete or rewrite original observations.

The current J/C report, historical eight-seed baseline, correction-subspace work, recursive composition, and systems studies are separate evidence families. A directory name or merged PR does not establish a general compression claim. The [older announcement checklist](docs/ANNOUNCEMENT_READINESS.md) still applies to its historical baseline scope.

The imported checkpoint is a bounded research record. New scientific work stays on separate branches until explicitly integrated; existing unrelated research PRs are not automatically merged. Generated scratch outputs belong under ignored `outputs/`, not primary evidence directories. Publication workflows use read-only permissions and do not authorize release tagging, external announcements or fresh experiments.
