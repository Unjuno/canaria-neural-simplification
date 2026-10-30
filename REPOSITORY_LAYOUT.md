# Repository layout

This map describes `research/checkpoint-20261026`, not a main release. Use the entry pages below to choose a route; the underlying code and evidence remain at their existing paths.

## Reading routes

| Goal | Entry point |
|---|---|
| Understand the project and its bounded results | [README](README.md) / [日本語](README.ja.md) |
| Read the scientific assessment and open questions | [Research assessment](RESEARCH_CHECKPOINT.md) |
| Find published reports and separate historical studies | [Research index](docs/RESEARCH_INDEX.md) |
| Check what has actually been validated | [Status](STATUS.md) |
| Recompute published scalar summaries without training | [Saved-result analysis](research/public_saved_analysis_20261027/README.md) |
| Inspect the separate historical eight-seed baseline | [Quickstart](QUICKSTART.md), [claim registry](docs/CLAIMS_AND_EVIDENCE.md), [reproduction discrepancy](docs/REPRODUCTION_DISCREPANCY.md) |

## Code, reports, and evidence

| Path | Role |
|---|---|
| [`src/canaria/`](src/canaria/), [`tests/`](tests/) | Reusable code and tests |
| [`scripts/`](scripts/) | Scientific runners, including historical baseline and supporting studies; read the associated protocol before use |
| [`research-checkpoints/`](research-checkpoints/) | Dated reader-facing reports and supplemental records; use the research index to distinguish scientific results from execution status |
| [`research/`](research/) | Research source packages, protocols, saved results, and scalar-recomputation tools; packages can require inputs not included in the repository |
| [`results/`](results/) | Original baseline, supporting studies, reproduction records, and active correction/invalidation evidence |
| [`docs/`](docs/) | Interpretations, limitations, corrections, and navigation, including historical baseline documents |
| [`environment/`](environment/), [`schemas/`](schemas/) | Environment records and data schemas |
| [`tools/`](tools/), [`publication/`](publication/) | Audits, scope policies, evidence catalogs, and preservation/migration manifests |
| [`local_evidence/`](local_evidence/) | Existing tracked native-cost evidence; the directory name does not make its contents private |
| [`archives/`](archives/) | Preserved historical protocols, code, outcomes, and former entry/review/release documents |

## Maintainability and workstreams (2026-10-31)

The [repository operations map](docs/REPOSITORY_OPERATIONS.md) identifies which checkout carries each research/publication role. The [workstream and open-PR index](docs/WORKSTREAMS.md) groups the 33 open PRs observed on 2026-10-31 without treating them as a unified merge queue. [Research package guide](research/README.md) and [dated checkpoints](research-checkpoints/README.md) distinguish source-only experiments from reports and failed/unfinished reproduction attempts. These links are navigation, not new science or a promotion to `main`.

## Preservation and boundaries

The [archive guide](archives/README.md) explains how to read relocated historical pages in their original context. The [migration map](publication/PATH_MIGRATION.json) records earlier moves. This map does not move or rewrite scientific records.

The current J/C report, the historical eight-seed baseline, correction-subspace work, recursive composition, and systems studies are separate evidence families. A directory name or merged PR does not promote a result into a general compression claim. The [older announcement checklist](docs/ANNOUNCEMENT_READINESS.md) belongs to the historical baseline, not the current report's validation status.

Only stable tests and intentionally supported workflows belong on the default branch. New research stays on its original branch and is linked by immutable commit. A new scientific release requires explicit evidence selection, not automatic merging of every draft PR.

Generated local outputs belong under ignored `outputs/`, not original evidence directories. Publication-validation uses read-only permissions and repeats existing evidence; it never launches fresh experiments or pushes code. New experiments, main promotion, releases, and social publication are separate from repository navigation.
