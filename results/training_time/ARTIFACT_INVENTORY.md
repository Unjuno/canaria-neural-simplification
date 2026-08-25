# Training-time artifact inventory

This directory contains compact public indexes for the training-time consolidation program.

## Files currently in the repository

- `summary.json` — G7–G17 headline results.
- `protocol_manifest.json` — seed ranges, protocol classifications, and protocol-integrity notes for the earlier training-time mainline.
- `late_stage_summary.json` — G18–G26 headline results plus recorded protocol/result SHA256 identifiers. G27 is exploratory and documented in `docs/LATE_STAGE_FINDINGS.md`.
- `README.md` — directory-level interpretation.

## What the SHA256 fields mean

A `protocol_lock_sha256` or `confirm_summary_sha256` records the identifier of the corresponding artifact used in the research session. It does **not** by itself imply that the raw artifact is present as a separate file in this repository.

Do not reconstruct a missing lock/result file from later summaries and then present it as contemporaneous evidence.

## Known late-stage limitation

During final public consolidation, the raw G20d/G20e protocol-lock files were not available in the retained final working snapshot. Their recorded SHA256 identifiers remain in `late_stage_summary.json`, and the result summaries were retained in the working archive. The missing raw locks are therefore treated as an artifact-retention limitation, not silently backfilled.

For several other G18–G26 phases, raw protocol/result files existed in the final working runtime but are not currently duplicated here as individual files. The public machine-readable index is `late_stage_summary.json`; future archival work may add original artifacts only when exact retained bytes are available.

## Evidence rule

The repository should distinguish:

1. **raw retained artifact present**;
2. **artifact hash recorded, raw bytes not currently public**;
3. **value reproduced from a later narrative summary only**.

Only (1) and (2) should be used for protocol-integrity claims. Category (3) is historical context, not a substitute for an original lock.
