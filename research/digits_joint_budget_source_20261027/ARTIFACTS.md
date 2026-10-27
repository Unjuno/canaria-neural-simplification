# Parent and grid artifacts — no payloads supplied

## New parent contract, not historical redaction

`PARENT_SCHEMA.json` describes a private descriptor produced by
`parent_contract.py`. It has exact fields, **339 exact artifact paths** and
**19 exact private custody paths**, each bound by SHA256. There is no included
real descriptor. Root/source/dependency paths are chosen by the reader; the
schema uses only fixed relative names. Symlinks, missing/extra files, changed
nested manifests and mismatched completions are rejected.

The original historical PARENT.json bound 339 historical output files including
`release.json`. Its SHA256 is retained in SOURCE_MAP.json; its contents and
private path are not exported. The new matched output uses `run_contract.json`
instead and keeps full approval in its durable private custody registry. The new
descriptor therefore checks **both** the entire output and that entire private
registry. It does not silently delete private binds from the historical chain or
claim the old verifier succeeded on a subset.

Parent output inventory and tensor/hash semantics are defined by the pinned
[matched artifact contract](../digits_matched_source_20261027/ARTIFACTS.md) and
[matched artifact schema](../digits_matched_source_20261027/ARTIFACT_SCHEMA.json).
The parent verifier hashes bytes without decoding arrays; runtime still checks
strict teacher state keys/shapes/dtypes/finiteness, initial/final state identities,
split arrays, data tensor hashes, (1077,64) float32 fit targets and int64
(600,128) schedule with indices 0…1076. It matches validation target/logits/labels
and teacher-correct count exactly to the verified parent. No teacher retraining
or 2048 checkpoint inference is used to manufacture reused cells.

The parent row's numeric seed record must equal the pinned matched SEEDS.json.
All L/J/C parent fit records must have 600 steps per parameter, 76,800 sampled
examples and the same schedule digest. Parent 2048 scalar accuracy and pass
flags are recomputed from the saved count/NMSE and copied verbatim for J/C.
Hash verification is integrity/custody evidence, not proof of a numerical training
trajectory; actual approvals are verified by the responsible custodian.

## New grid output

```text
GRID_CUSTODY_ROOT/
  custody/grid/<replication_id>/     # private, permanent, never publish raw
    claim.json, cohort.json, FINISHED.json, seed_<id>.started.json
  artifacts/<replication_id>/
    run_contract.json, SUMMARY.json, FILES_SHA256.json
    seed_<id>.log, seed_<id>_EXIT.json
    seed_<id>/
      START.json, environment.json, result.json, FILES_SHA256.json
      <arm>_<budget>_initial.npz, <arm>_<budget>_final.npz
      <arm>_<budget>_START.json, <arm>_<budget>_updates.jsonl
      validation_arrays.npz
```

For each ID 0…15, arm is J or C and new budget is one of
512,1024,1536,3072,4096,6144. There are no new 2048 weight/update artifacts.
Each update log has 600 rows. J states contain four float32 bias-free matrices
for two TinyRes maps of width h; C contains two matrices for width 2h, where
budget=256h. Runtime validates loading of teacher states, not arbitrary pickle
payloads (`allow_pickle=False`).

Validation arrays contain float32 target (270,64), teacher logits (270,10),
int64 labels (270,), and endpoint (270,64)/logits (270,10) for each of twelve new
arm-budget cells. The 14-cell per-teacher result includes two identified reused
2048 cells. SUMMARY records all224 cells, first-observed-pass budgets, coverage,
per-budget pass counts, nonmonotonic patterns and gated bootstrap decision.
Any failure leaves its available evidence and FAILURE/exit/not-started census;
there is no selective completion or replacement.

Complete successful output has 883 files including its final root manifest
(882 entries in that root manifest): 53 per-seed files ×16 plus35 root files.
This count documents layout, not a validation result for any new run. Root and
seed FILES_SHA256 maps exclude themselves. Tensor hashes, state hashes and file
hashes are different identities; NPZ bytes need not match if tensors match.

Generated logs, errors, approval evidence, paths and process/custody information
are not automatically public-safe. Source publication does not authorize artifact
publication. No model/data/license custody is inferred from public scalar results.
