# Repository layout

The current entry points are [README](README.md), [Quickstart](QUICKSTART.md), [claim registry](docs/CLAIMS_AND_EVIDENCE.md), [research index](docs/RESEARCH_INDEX.md), and [readiness gate](docs/ANNOUNCEMENT_READINESS.md).

| Path | Role |
|---|---|
| `docs/` | Current baseline interpretation, corrections, limitations and research navigation |
| `results/core_discovery_digits/` | Original headline protocol and numerical summary |
| `results/replication/`, `training_time/`, `phase2/` | Supporting baseline evidence and correction status |
| `results/reproduction/` | Repeat-run/portability evidence, never extra fresh confirmation |
| `scripts/reproduce/`, `replication/`, `phase2/` | Baseline scientific runners |
| `scripts/phases/training_time/` | Supporting training-time scripts |
| `src/canaria/`, `tests/` | Small reusable-code surface and tests |
| `tools/`, `publication/` | Audits, scope policy and byte-preservation/migration manifests |
| `archives/` | Historical protocol/code/result/review/release material |

The old versioned research sequence is archived without changing its source bytes. [Migration map](publication/PATH_MIGRATION.json)

New research stays on its original branch and is linked by immutable commit. Original prospective histories, negative results and unresolved experiments survive. Updating navigation is not approval of a research claim. A new scientific release requires explicit evidence selection, not automatic merging of every draft PR.

Only stable tests and intentionally supported workflows belong on the default branch. Publication-validation repeats existing evidence; it never launches fresh experiments or pushes code. It uses read-only permissions. Generated local outputs belong under ignored `outputs/`, not the original evidence directories.
