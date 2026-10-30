# Required private handoff and output meanings

For the separate fresh-replication input schema and reviewed selection/storage/
saved-reference handoff, see [FRESH_INPUTS.md](FRESH_INPUTS.md). The historical
v1 requirements below are unchanged and are not relaxed by fresh mode.

No data/model/log/registry payload is included. The code cannot run a historical
model experiment from this package alone. Source closure is not payload release,
full historical operational replay, independent replication or new permission.

## Input manifest v1

All paths in the new manifest are canonical relative POSIX names contained in one
explicit, already-authorized local input root. Absolute paths, URLs, traversal,
backslashes, symlinks, duplicate JSON keys and unknown descriptor fields are rejected.
No network fetch, installed-dataset search or alternative-source fallback exists.
JSON shapes are declared by INPUT_SCHEMA.json; the runtime additionally enforces
cross-file SHA256, cohort, choice and reference bindings.

Required descriptors contain `path` and `sha256`:

- `selection`: portable `canaria.selected-whole-storage.v1` selection manifest,
  with exactly 16 IDs, 48 distinct source NPZ paths and fixed C/J budgets/hashes.
  The byte-identical storage validator is vendored as storage_manifest_contract.py.
- `origin_selection`: original immutable storage selection bytes, with the pinned
  original SHA256. Their private path strings are lookup/provenance data only.
- `historical_reference_manifest`: the original pinned 47-entry reference manifest
  (16 split archives, 30 distinct validation archives, one reference selection).
  Its old paths are never resolved or opened by this package.
- `reference_selection`: the separate pinned original reference-choice record.
- `csv`: explicitly authorized local `digits.csv.gz` byte stream with the frozen
  SHA256, never downloaded. Only validation feature rows/columns are materialized.
- `storage`: `directory` of stored whole-model archives plus `artifacts` and `rows`
  descriptors pointing to ARTIFACTS.json/ROWS.json. ARTIFACTS lists all 96 original
  stored/deflated siblings; latency reads and requires only the 48 stored archives.
  Every stored byte size/hash and all 16 parameter-count rows are checked.
- `validation`: 16 objects with `id`, `split` descriptor, and exactly teacher/C/J
  `logits` descriptors, each additionally carrying `key`.

Custodian relocation must preserve every choice. The loader compares IDs, teacher
file hashes, both arm budgets/file hashes/logical hashes with both frozen selection
records; it checks each split and selected logit SHA256 against that ID's original
reference-manifest lookup. Budget 2048 uses the matched archive keys `C_logits` or
`J_logits`; other budgets use `C_<budget>_logits` or `J_<budget>_logits`.
Teacher uses `teacher_logits`. Paths alone may change; no reselection or new arrays.

The original reference-manifest bytes and original selections remain PRIVATE
because their strings contain operational paths. They are not bundled publicly.
Independent custody review remains necessary for permission, completeness, actual
file provenance and the authorized path relocation. The new validator is not
claimed to execute the original historical release/registry/clean-tree checks.

## Release and single attempt

RELEASE_SCHEMA.json binds input SHA256, all package hashes (including SOURCE_MAP),
clean implementation commit, authorizer-approved attempt ID/control root, explicit
new deadline and human review attestations. `control_root_sha256` is SHA256 of the
canonical absolute control-root path encoded in UTF-8. The path itself need not
appear in the release. The shared root must remain stable across its approval
scope. A new release/attempt must not be manufactured to repeat a failed result.

## Generated outputs are PRIVATE by default

- RAW.jsonl: all 2,880 ordered blocks; OUTPUT_SCHEMA.json describes one row.
- CENSUS.json: all 96 seed/batch/arm cells, including failed/not-started cells.
- CORRECTNESS.json: all 48 graph checks, max errors and input-relative references.
- INPUT.npy: the synthetic timing input, not a dataset or model.
- SUMMARY.json: per-cell medians, all paired ratios and aggregates, or failure.
- RELEASE.json: new private authorization metadata; never the original release.
- PRIVATE_ENV_BEFORE/AFTER.json: raw CPU/affinity/load/process/build metadata.
- PRIVATE_FAILURE_TRACEBACK.txt: private diagnostic traceback if one is available.
- Permanent registry: REQUEST/START/CENSUS/EXIT/SUPERVISOR_EXIT JSON, private.

The supervisor never calls an exit-zero partial worker complete: it requires 96
complete cells, complete worker status, exactly 2,880 blocks, and valid terminal
metadata. Missing/corrupt worker metadata yields a conservative failed census.
Evidence I/O failure or an unreaped worker is not recoverable success. Review
before any further authorization; no automatic retries or cleanup exist.

Neither public source packaging nor any later synthetic-test success authorizes
publication of these generated outputs. Raw identity/process/path material,
original private manifests, weights and datasets must remain excluded.
