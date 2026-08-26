# Archives

This directory preserves research provenance that should remain inspectable without occupying the current evidence/navigation surface.

**Archived does not mean false.** It means the material is historical, superseded as current guidance, or part of an older version-number research sequence. Scientific status must still be read from the relevant protocol/result/correction record.

## Current archive groups

### `research-history/`

Preserves the older iterative experiment sequence that previously appeared directly under `docs/`, `results/`, `scripts/`, and `environment/`.

Migration map:

| Previous active path | Archived path |
| --- | --- |
| `docs/history/` | `archives/research-history/docs/history/` |
| `docs/phases/` | `archives/research-history/docs/phases/` |
| `results/history/` | `archives/research-history/results/history/` |
| `results/phaseA_v11/` | `archives/research-history/results/phaseA_v11/` |
| `results/phaseB_v11/` | `archives/research-history/results/phaseB_v11/` |
| `results/v12/` … `results/v25/` | `archives/research-history/results/v12/` … `results/v25/` |
| `scripts/phases/v11/`, `v17/` … `v23/` | `archives/research-history/scripts/phases/...` |
| `environment/history/` | `archives/research-history/environment/history/` |

Older broad planning/status documents are under `archives/research-history/legacy-docs/`.

The move was performed for pre-announcement readability under Issue #16. The historical trees/blobs were preserved rather than rewritten.

### `reviews/`

Historical review handoffs that are no longer active instructions. The completed pre-publication handoff formerly at repository root is preserved here.

### `releases/`

Historical release/snapshot gate documents. These describe a past repository boundary and must not be used as the current announcement-readiness gate.

The immutable `v0.2.0-public-snapshot` tag remains the authoritative frozen Git boundary for that snapshot.

## How to read archived experiments

For an archived experiment, read together:

1. the archived protocol/narrative;
2. the archived machine-readable result;
3. the matching archived runner when available;
4. any later current correction/invalidation record.

Do not use a high version number as a proxy for evidential strength. Some late experiments are exploratory, negative, uncertain, or superseded.

Archived documents may contain stale relative links because preserving their original text is part of provenance. Use this index as the migration map instead of editing historical documents in place.

## Snapshot checksum provenance

`canaria_public_research_snapshot.zip.sha256` is retained as historical archive provenance. It does not define the current repository surface.
