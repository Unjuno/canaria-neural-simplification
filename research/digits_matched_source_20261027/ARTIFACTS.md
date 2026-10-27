# Relative artifacts and custody boundary

`ARTIFACT_SCHEMA.json` validates a future `FILES_SHA256.json` path-to-digest map.
Its `x-*` annotations describe the exact payload inventory and array shapes;
those annotations are documentation, **not an executable tensor validator**.
No arrays, checkpoints, models, logs or private historical release are supplied.

For a separately approved replication, the operator chooses one durable workspace:

```text
workspace/
  custody/matched/<replication_id>/   # PRIVATE; permanent, one-attempt registry
  artifacts/<replication_id>/       # NEW results; review before any distribution
    run_contract.json
    SUMMARY.json
    FILES_SHA256.json
    seed_0.log ... seed_15.log
    seed_0_EXIT.json ... seed_15_EXIT.json
    seed_0/ ... seed_15/
```

A successful seed writes 19 files including its own hash manifest. Failure can
leave a smaller prefix plus `FAILURE.json`. Failed, not-started and nonzero-exit
seeds remain in the cohort census; they are not replaced. Root logs and failure
reasons can contain local error paths, so generated results are **not automatically
public-safe**. The public source package contains none of them.

Teacher state names are `stem.{weight,bias}`, `blocks.<0..3>.ln.{weight,bias}`,
`blocks.<0..3>.fc1.{weight,bias}`, `blocks.<0..3>.fc2.{weight,bias}`, and
`head.{weight,bias}`. Stem is 64→64; each residual block is LayerNorm(64),
64→128→64 with GELU; head is 64→10. L/J matrices are `0.w1.weight` (8×64),
`0.w2.weight` (64×8), `1.w1.weight` (8×64), `1.w2.weight` (64×8).
C has `w1.weight` (16×64) and `w2.weight` (64×16). All are float32;
TinyRes maps are bias-free residual maps.

Tensor identities are SHA256 of concatenated torch dtype text, shape-tuple text
and contiguous CPU NumPy bytes. State identities are SHA256 of a sorted JSON
mapping from tensor names to tensor digests (default JSON separators, as in the
runner). File SHA256 hashes are separate from tensor/state hashes. ZIP container
bytes need not match merely because tensor identities match.

Split arrays expose only fit and validation IDs. The loader reads full bundled
Digits features/labels for stratification, discards the test IDs and never
materializes or evaluates test subsets. Validation arrays are saved only after
all three replacements finish. Validation is already seen data; excluding test
arrays does not restore fresh-data status to any earlier test.

The original model/tensor archive is not reconstructible from scalar results or
hashes alone. A new approved execution produces new files under this contract,
not the original missing artifact bytes. This contract is not compatible with
the original grid's private parent-artifact/coordination manifest; do not remove
its missing entries and claim that the original parent verifier passed.
