# Input contract and safe selection handoff

## What is deliberately missing

Historical `SELECTION.json`, all 48 weight NPZ files, all 96 serialized archives,
raw results and audits, datasets, host/process metadata and operational records
are not included. The source package contains no real or synthetic payload files;
artificial arrays exist only as test generation code. The historical selection
manifest's digest in `SOURCE_MAP.json` identifies its original bytes but does not
release those bytes or grant access.

Storage cannot independently reproduce historical bytes without authorized exact
inputs. Rebuilding new matched/grid models would be a separate scientific run,
not a source-packaging or storage-only task. No inference, fitting or historical upstream
manifest-generation helper is provided here. The separate [fresh-replication
handoff](FRESH_SELECTION.md) consumes completed fresh outputs only; it does not
recover or relabel this historical archive.

## Portable manifest v1

`SELECTION_SCHEMA.json` declares the JSON shape; `manifest_contract.py` additionally
checks exact fields, duplicate JSON keys, 16 unique integer IDs 0–15, all 48 unique
safe relative NPZ paths, hexadecimal hashes and admitted budgets. Boolean IDs and
budgets are rejected. All keys are required and unknown fields are rejected.

The object has:

- `schema`: exactly `canaria.selected-whole-storage.v1`.
- `cohort_size`: integer 16.
- `selection_origin_sha256`: SHA256 of the custodian's immutable original selection
  record. This is a provenance claim, not a value the source-only validator can
  establish from unpublished scientific records.
- `models`: 16 objects with `id`, `teacher`, and `arms`.
  - `teacher`: `path`, `file_sha256`.
  - `arms`: exactly `C` and `J`, each with `path`, `file_sha256`, `state_sha256`,
    and integer `budget` from the frozen seven-budget set.

Paths are canonical relative POSIX `.npz` names, e.g.
`inputs/seed_0_teacher.npz`. Absolute paths, traversal, URL syntax, backslashes,
empty components, symlink components and files outside the input root are rejected.
Source files are not renamed inside their NPZ tensor keys. Resolving an input root
is a user-supplied local custody choice, never an embedded historical location.

## Historical custodian handoff — must precede an authorized storage run

1. Retain the exact historical selection record privately and verify its recorded
   hash. Do not rescore models, choose new thresholds, optimize compression,
   drop failures, replace IDs, or recalculate selection under this handoff.
2. Copy only the 48 already-selected NPZ byte streams into a separately approved
   custody directory. Read permission and model/data redistribution permission
   are separate; preparing a local manifest does not authorize publication.
3. For each ID, preserve the chosen teacher file SHA256 and each C/J budget,
   file SHA256 and logical-state SHA256. Relocate only file paths. Retain a private
   one-to-one original-path/new-path map; do not publish the operational map.
4. Verify source file bytes against the frozen record. The selected replacement
   logical hash is computed per key as SHA256 of
   `b'torch.float32' + str(tuple(shape)).encode() + array.tobytes(order='C')`;
   the model logical hash is SHA256 of `json.dumps(per_key_hashes, sort_keys=True)`.
   This preserves the historical string convention without importing PyTorch.
5. Emit the v1 manifest and calculate its own SHA256. Independent review must
   compare all 16 IDs and all 32 selected arm choices with the frozen record;
   only then supply that digest to `--manifest-sha256`. A digest copied from an
   unreviewed changed manifest does not validate its selection decisions.
6. Only after separate execution authority, run the explicit CLI into a fresh
   outside-tree output directory. Keep generated artifacts private by default;
   publication requires a separate payload review.

The upstream rule was the smallest passing budget among the fixed grid, where
`nmse <= 0.08` and `correct/270 >= teacher_correct/270 - 0.02`; budget 2048 reused
the earlier matched cells. The selected 2048 weights therefore come from the
matched outputs, other selected budgets from the grid outputs. This describes
provenance only. The source-only validator does not re-evaluate that rule, load
validation arrays, or prove the original selected choices from model hashes.

## Output meanings and failure policy

- `ROWS.json` / `ROWS.csv`: one completed teacher row with entire-model counts,
  archive byte sizes and nine ratios.
- `CENSUS.json`: all 16 IDs, PASS or FAIL. Failure records carry exception types,
  not raw exceptions/tracebacks that could leak local paths.
- `ARTIFACTS.json`: every produced archive filename, filesystem bytes and SHA256.
- `SOURCES.json`: input-relative source identity records; no absolute host paths.
- `TENSORS.json`: source/destination tensor mapping, shape, dtype and raw bytes hash.
- `SUMMARY.json`: fixed cohort, complete/failure counts, 96-archive census,
  aggregate ratios and COMPLETE/INCOMPLETE status.
- `ENV.json`: only Python/NumPy/zlib version and source/manifest hashes. No host,
  process command, local path, Git working-tree lookup or operational token.

An exception may leave partial sources/mappings/archives for its teacher, exactly
as an incomplete run requires evidence rather than deletion. Such partial files
are not silently promoted into completed rows. Existing output is always refused;
no automatic retry, cleanup, resume or recovery is implemented. Review a failure
before authorizing any new attempt. Abrupt termination can prevent final metadata;
absence of a complete census always means incomplete, never an implicit pass.
