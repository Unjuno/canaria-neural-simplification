# Fresh matched/grid → storage selection handoff v1

This is a **fresh-replication provenance contract**, not recovery or relocation of
the original historical archive. It consumes only already-completed, separately
authorized fresh matched and grid outputs from the pinned sibling sources.
Source publication does not authorize reading models, copying payloads, scientific
execution, storage measurement, publication of payloads, or a retry.

`prepare_fresh_selection.py` is a bounded helper beside the unchanged storage
reader. It does not train, infer, choose thresholds, fit new budgets, rerun a
bootstrap, select checkpoints, rewrite NPZ tensors, or change historical results.
It checks the already-fixed first-observed-pass rule; it makes no new scientific
choice. Generated files are private and **PENDING_CUSTODIAN_REVIEW**.

## Inputs and separate review boundaries

An authorized custodian supplies:

1. A completed fresh matched custody workspace and exact private descriptor from
   the sibling `parent_contract.py`. The helper verifies all 339 artifact and 19
   custody files and the original approval/release/deadline bindings.
2. A separate completed fresh grid workspace and `replication-…` identity. The
   helper verifies all 883 artifact and 19 custody files, full and nested hash
   manifests, all 16 worker admissions/exits/results, source identities, private
   release, claim, completion stamp and existing deadline. A complete-looking
   summary without complete underlying files is insufficient. An expired but
   timely-completed attempt is not revived; late completion is rejected.
3. A **separately reviewed**, immutable private input-review JSON, pinned by its
   SHA256 out of band. Its exact required keys are:

   | Key | Required value |
   |---|---|
   | `schema` | `canaria.fresh-selection-input-review.v1` |
   | `purpose` | `fresh-grid-to-storage-only` |
   | `matched_descriptor_sha256` | SHA256 of the exact descriptor bytes |
   | `grid_replication_id` | The exact fresh grid identity |
   | `grid_files` | All 883 relative artifact paths → SHA256 |
   | `grid_custody_files` | All 19 relative custody paths → SHA256 |
   | `adapter_files` | Exact SHA256 map for `prepare_fresh_selection.py`, `SELECTION_DEPENDENCY.json`, `storage_codec.py`, `manifest_contract.py` |
   | `upstream_provenance_approved` | JSON `true`, only after actual review |
   | `independent_review` | `GO`, only after actual independent review |
   | `reviewer`, `approval_evidence` | Nonempty private provenance references |

   Maps are obtained from a read-only full inventory and compared independently
   with the actual completed custody chain. This helper does **not** generate a
   GO record or infer approval from a digest. Review/GO strings are attestations,
   not signatures or cryptographic proof. The synthetic test fixture's GO words
   never authorize an actual input or execution.

The full sibling matched/grid source tree is pinned by
`SELECTION_DEPENDENCY.json` at public base
`dd8b8e012bcf9acdb3f78641062b2c8343394d99`. Only stdlib parent/grid contract
modules are loaded, never either numerical runner. The unchanged storage codec
is loaded only after all upstream and input-review checks, solely for strict
selected-array validation and the existing logical-state hash convention.

## Exact selection and source routing

All 224 unique cells must be present. Teacher identities, parent result hashes,
initial/final teacher state hashes, shared schedule hashes, fixed fit records,
counts, accuracy, finite NMSE, boolean quality flags and reused-cell identities
are checked. Result/summary and exit/summary cross-bindings use recursive
type-sensitive JSON equality: `false`, `0` and `0.0` are not interchangeable.
Result, exit, custody-admission and worker-START IDs must be exact JSON integers,
and the START seed map must also match without numeric-type coercion. A pass is exactly `nmse <= .08` and
`correct/270 >= teacher_correct/270 - .02`. The helper compares all 16 existing
minima, seven-budget pass counts, nonmonotonic patterns and coverage to those
deterministically derived from the cells. There is no interpolation or
monotonicity assumption. The original summary—including its statistical outcome
and all rows—is retained exactly as a JSON value and also bound by its original
file SHA256; bootstrap statistics are not recalculated or used to change choices.

If **any** teacher/arm is censored, no manifest or copied-source output is
created. The upstream censored record is not changed; no teacher is dropped,
substituted or assigned an invented budget.

For each of 16 teachers:

- teacher source: matched `seed_<id>/teacher_final.npz`;
- selected budget 2048: matched `seed_<id>/<C-or-J>_final.npz`;
- other selected budgets: grid `seed_<id>/<C-or-J>_<budget>_final.npz`.

All 48 selected files are checked for exact file SHA256, archive member census,
strict float32 shapes/keys, finiteness and logical-state SHA256, including teacher
state identity. Sources are copied byte-for-byte, not reserialized. Teacher
logical hashes remain in fresh provenance because unchanged storage v1 has no
teacher logical-hash field. All other upstream files are fully hash-bound, not
numerically decoded; these checks do not prove truthful numerical trajectories
or validate dataset/model science independently of upstream custody review.

## Authorized preparation command shape

Only after separate authority for these exact inputs and local copying:

```sh
python -B prepare_fresh_selection.py \
  --matched-workspace /authorized/fresh-matched \
  --matched-descriptor /authorized/reviewed-parent.json \
  --grid-workspace /authorized/fresh-grid \
  --grid-replication-id replication-approved-fresh-grid \
  --input-review /authorized/reviewed-inputs.json \
  --input-review-sha256 <independently-approved-review-sha256> \
  --output /authorized/absent-private-selection
```

These are placeholders, not supplied data or an execution authorization. The
output parent must exist; output must be absent, nonsymlink, and outside all
input/source trees. Inputs are checked again before output creation. Failure
before creation leaves no output. I/O failure after creation may leave partial
files: retain them for review; never treat them as a complete bundle. There is
no overwrite, resume, cleanup or automatic retry.

## Output and mandatory downstream gate

A successful preparation writes exactly:

- `inputs/seed_<id>_<teacher-or-C-or-J>.npz`: 48 exact source byte streams;
- `FRESH_SELECTION.json`: `canaria.fresh-grid-storage-selection.v1`, complete
  upstream descriptor and grid file/custody maps, original grid summary, input
  review digest, source pins, choices and one-to-one source/destination map;
- `selected_models.json`: unchanged storage v1 schema; `selection_origin_sha256`
  binds the exact fresh `FRESH_SELECTION.json` bytes, **not** the historical
  original selection digest;
- `REVIEW_REQUIRED.json`: pending status, manifest/origin digests, count 48 and
  `storage_execution_authorized: false`.

**An independent custodian must next compare all 16 teachers and 32 arm choices,
all source and logical hashes, the fresh provenance and exact copied bytes.**
Record the review externally and separately authorize any subsequent storage
execution. The adapter never outputs an approved review, launches `measure.py`,
or turns a manifest digest into authorization. Existing `measure.py` validates
manifest structure, not this human gate; passing the generated digest directly
to it before review would bypass the required operational gate and is not an
authorized workflow. Historical custody instructions remain separate.

This does not close the fresh storage → latency provenance/input-bundle gap.
Historical latency v1 pins and its private inputs remain unchanged. It is not an
end-to-end executed fresh pipeline, a historical replication, or a new result.

## Focused artificial verification

After approval for bounded local test execution:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -B test_fresh_selection.py
```

The suite creates temporary artificial arrays and producer-shaped metadata only.
It covers every selected budget, 2048 routing, exact deterministic copies,
nonmonotonic patterns, censoring, missing/extra/corrupt files, false pass/selection
claims, failed workers, late completion, absent review, source pins, duplicate
JSON/IDs/cells, symlinks and output boundaries. No real weights, dataset, original
NPZ, ML runner, training/inference, timing, storage measurement or scientific
attempt is used. It does not rerun the pre-existing package suites.
