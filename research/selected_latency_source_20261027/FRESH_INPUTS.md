# Fresh storage → latency handoff, version 1

This is a separate **fresh-replication input contract**, not historical reproduction
or a replacement for historical input v1. `latency_contract.py`, its three original
historical provenance digests, the frozen CSV digest, `INPUT_SCHEMA.json` and
`RELEASE_SCHEMA.json` are unchanged. No historical registry is reopened.

This source-only revision supplies `fresh_latency_contract.py` and a metadata-only
`prepare_fresh_latency.py`. It supplies **no input instance, model, saved array,
review GO, release, manifest payload, registry or execution authority**. A local
source candidate and artificial tests do not establish a scientific replication.
All original consumed-attempt, UCI, no-retry and individual holds remain in force.

## Existing references, and what is missing

The public matched producer saves `seed_<id>/split_ids.npz` (`validation`, 270
indices) and `validation_arrays.npz`. The latter contains `teacher_logits`,
`C_logits`, `J_logits` (plus other arrays not read by this verifier). Teacher and
selected C/J **budget 2048** must use these matched references.
The public grid producer saves its own `validation_arrays.npz`, containing
`C_<budget>_logits` and `J_<budget>_logits` for all six non-2048 budgets. It does not
save newly evaluated 2048 arm logits: those references remain matched-only.
Each required logit is float32, finite, shape `(270, 10)` in the original saved
validation row order. The exact original NPZ bytes, not a reserialization, must
match the fresh selection provenance's matched/grid file inventories.

There is **no fixed 47-file fresh reference census**. There are always 16 split
files, 16 matched validation files and one grid validation file for each teacher
with at least one selected non-2048 arm (0–16), although the 48 teacher/arm
reference descriptors repeat files when appropriate. The reference mapping is
derived deterministically from the 32 already-fixed budgets; it is not selection.

Neither source packages nor scalar reports reconstruct these arrays. This work
inspects only producer source and uses explicitly artificial fixtures; it has not
located, opened or validated any actual fresh-run references. Missing files,
missing keys, wrong hashes/dtypes/shapes or unreviewed provenance fail closed.
There is no inference/preparation mode that manufactures substitute references.
The separately authorized local `digits.csv.gz` must still match the unchanged
frozen SHA256. It is not supplied by this adapter and is never downloaded.

## Private input layout and exact contract

Use one explicit existing authorized input root. Every descriptor is exactly
`{path, sha256}`, plus `key` only for logit references. Paths are canonical
relative POSIX names; no URL, absolute path, traversal, symlink (including root
ancestors), duplicate JSON key or nonfinite JSON number is accepted.
`fresh_latency_contract.validate_inputs` is the authoritative exact-fields schema:

- `schema`: `canaria.selected-latency-fresh-inputs.v1`;
- `selection_directory` and descriptors `selection`, `fresh_selection`,
  `pending_gate`: original `selected_models.json`, `FRESH_SELECTION.json`,
  `REVIEW_REQUIRED.json` from the published selection helper. Their bytes and
  the 48 `inputs/seed_<id>_<teacher-or-C-or-J>.npz` sources stay unchanged;
- `parent_descriptor`, `upstream_review`: the exact independently reviewed
  matched descriptor and selection-input-review bytes bound by FRESH_SELECTION;
- `matched_directory`, `grid_directory`: roots holding the original reference
  NPZ files; `grid_summary`: original grid `SUMMARY.json`, hash-bound to the
  full upstream inventory and type-exactly equal to the embedded original summary;
- `storage`: `directory` plus exact `files` descriptors for `ENV.json`,
  `ROWS.json`, `CENSUS.json`, `ARTIFACTS.json`, `SOURCES.json`, `TENSORS.json`,
  `SUMMARY.json`, `ROWS.csv` from the completed unchanged storage producer;
- `csv`: explicitly authorized local frozen CSV descriptor;
- `validation`: ordered IDs 0–15, each with `split` and exactly teacher/C/J
  `logits` descriptors, derived by `validation_descriptors` above;
- `input_review`: exact approved external fresh handoff review descriptor.

No original upstream registry is needed at the latency host. Complete matched
(339 artifacts/19 custody files) and grid (883/19) inventories remain in the
immutable selection record. Their original completion/source/custody verification
is established by the separately reviewed selection helper and input review,
**not rerun or magically proven by an inventory hash at this boundary**. This
adapter pins all 53 sibling matched/grid/storage source files, verifies the
selection helper identity, all 224 cell identities/quality flags, original minima,
nonmonotone patterns and coverage, plus every selected source/state binding.
It preserves the original grid decision and refuses censoring or missing teachers.
It does not recalculate bootstrap statistics or assert truthful trajectories.

## Mandatory independent handoff review

The selection helper's `PENDING_CUSTODIAN_REVIEW` / false authorization records
remain immutable. They are **not** sufficient for downstream use. A separate
reviewer must approve the 16 teachers/32 choices, source/whole-model content,
original saved-reference provenance and prior storage execution authority.
The external review has exactly these fields:

| Field | Required binding |
|---|---|
| `schema` | `canaria.selected-latency-fresh-review.v1` |
| `purpose` | `fresh-storage-to-latency-only` |
| `input_basis_sha256` | SHA256 of canonical `encoded(manifest without input_review)` |
| `source_manifest` | Exact `fresh_latency_contract.package_hashes` for this reviewed source |
| `selection_review` | `GO`, after independent comparison, never copied from pending gate |
| `storage_execution_approved` | JSON `true`, attesting separately granted prior storage authority |
| `storage_complete_review` | `GO`, after complete 16/48/96 storage review |
| `reference_provenance_review` | `GO`, after verifying genuine saved upstream references |
| `independent_review` | `GO` |
| `reviewer`, `approval_evidence` | Nonempty private provenance references |

These fields are human attestations, not signatures, execution permission, or
proof of truthful numerical output. The helper never writes this review or turns
a digest into approval. The read-only `describe_inputs` function exposes the
exact candidate input basis to an authorized reviewer; it does not issue a GO.
Relocation changes the basis and requires review of the relocated paths/bytes.

## Preparation and separately authorized entry path

After separate authority for the exact input files and manifest creation, use
`prepare_fresh_latency.py` with explicit `--input-root`, `--selection-directory`,
`--storage-directory`, `--matched-directory`, `--grid-directory`,
`--parent-descriptor`, `--upstream-review`, `--csv`, `--input-review`,
`--input-review-sha256` and `--output`. All paths except input root and output
are relative to input root. Output must be a new absent file outside the input
and source trees. The helper hashes metadata, sources and required stored/reference
bytes twice, then creates only that private manifest. No model/array decoding,
copying or numerical imports occur. A partial failed write is retained; no
cleanup, overwrite, resume, selection change or retry is automatic.

Only the original `supervise.py` may start a worker, with the existing required
arguments and explicit `--input-mode fresh-v1`. This mode requires a new release
schema `canaria.selected-latency-fresh-release.v1`; all other release fields,
source/input/commit/control-root bindings, approval attestations and one-attempt
lifecycle checks are exactly the historical portable release contract. Its
purpose remains `separate-authorized-technical-attempt-not-historical`.
The original `historical-v1` is the default and accepts only its original input
and release schemas. Neither mode falls back to the other.

The fresh worker lazily checks source/state hashes, reconstructs whole-model
arrays with the unchanged storage codec, compares every stored tensor byte,
parameter/payload count and TENSORS record, and checks only the selected saved
logit/split arrays. The original deployment loader then performs the unchanged
48 graph correctness tests (270 rows, chunks 256+14, maximum error 1e-5) before
any timing. No label array or held-out test is materialized by this fresh verifier.

`deployment.py`, `model_definitions.py` and `timing_core.py` remain byte-identical.
So do the timing loop's AST, model/seed/batch order, input seed 20261027,
20 warmups, 30 rounds ×20 calls, literal six permutations, 96-cell/2880-block
census, all paired summaries and no-exclusion policy. The same supervisor retains
nonblocking shared locks, permanent one-attempt claim, 590+5+5-second envelope,
bounded TERM/KILL/reap, 4GiB worker cap and 128MiB evidence cap/rescue reserve.
Fresh mode is forwarded to the worker and separately release-bound; it grants
no new attempts, no retry, no deadline extension and no historical resume.

## Artificial proof only

The focused `test_fresh_latency.py` suite uses newly created labelled artificial
arrays/metadata and fake clocks/callables, never original scientific payloads.
Positive bundle tests narrowly fake the digest of one labelled artificial CSV
file; **the unmodified frozen-CSV gate rejects that file without the test mock**.
No historical constant is changed. Artificial saved logits are fixture-only and
are never represented as actual validation, correctness evidence or performance.
Negative tests exercise absent reviews, provenance/choice/reference/byte/state/
content/census drift, historical/fresh mode separation and output boundaries.
The unchanged `test_controls.py` exercises fake timing and cleanup; no real model
forward, dataset access, training, benchmark timing, network, installation or
scientific attempt occurs. Passing tests establish control behavior only.

Every actual generated input/review/output/registry remains PRIVATE by default.
This source task authorizes neither real execution nor publication of any payload.
