# Selected whole-model storage — portable source-only package

**Status: source-only candidate, statically inspected; no execution or scientific
reproduction is claimed.** No historical weights, data, selected-model manifest,
archives, results, logs, or operational records are included. The historical
measurement is complete elsewhere; this package does not authorize another run.

This package extracts the fixed storage algorithm into an explicit CLI without
import-time output creation. It deliberately does **not** implement training,
selection generation, inference, latency, deployment, network access, or dataset
loading. Its sole numerical dependency is NumPy.

## Files and dependencies

- `measure.py`: mandatory-manifest, explicit fresh-output storage launcher.
- `storage_codec.py`: deterministic serializer, strict source reader and exact
  whole-model assembly. This module imports NumPy but reads/writes nothing on import.
- `manifest_contract.py` and [SELECTION_SCHEMA.json](SELECTION_SCHEMA.json):
  dependency-free manifest/containment validator plus declarative schema.
- `test_storage.py`: **artificial arrays only**, supplied for separate review.
- [PROTOCOL.md](PROTOCOL.md): fixed serialization and denominator semantics.
- [ARTIFACTS.md](ARTIFACTS.md): missing input contract and safe custody handoff.
- [ENVIRONMENT.md](ENVIRONMENT.md): exact-version recipe and known build gaps.
- [TRANSFORMATION.md](TRANSFORMATION.md) and [SOURCE_MAP.json](SOURCE_MAP.json):
  historical-source identities, changes, and export hashes.
- [STATIC_VALIDATION.json](STATIC_VALIDATION.json): what static inspection checks,
  and, explicitly, what it does not establish.

No runtime dependency on the sibling
[matched source](../digits_matched_source_20261027/README.md) or
[grid source](../digits_joint_budget_source_20261027/README.md) is imported. Those
packages explain the upstream scientific chain; their publication is not evidence
that this package has all historical model payloads.

## Authorized use after separate review and input custody approval

1. Obtain approval for the particular storage run and the exact 48 input files.
   This source release is not that approval. Do not reactivate historical jobs,
   rewrite their registries, or treat source availability as data/model release.
2. A custodian prepares an immutable `selected_models.json` according to
   `ARTIFACTS.md`; no selection is performed here. Pin its SHA256 out of band.
   The manifest must bind all 16 teacher IDs, selected C/J budgets, source file
   hashes and replacement logical-state hashes, with relative NPZ paths only.
3. Use the exact existing environment described in `ENVIRONMENT.md`. Installation
   or downloading is a separate action and was not performed for this package.
4. From the package directory, the command shape is:

   ```sh
   python measure.py \
     --manifest ../../authorized-inputs/selected_models.json \
     --manifest-sha256 <approved-manifest-sha256> \
     --input-root ../../authorized-inputs \
     --output ../../authorized-storage-output-new
   ```

   Substitute the approved digest; angle-bracket text is not an executable value.
   Paths are examples, not provided inputs. The output parent must exist. The
   output itself must be absent, outside both the input and package trees.
   Existing output is refused **before manifest reads and NumPy import**; nothing
   is deleted, merged, resumed, or overwritten. There is no default `results/`.
5. Require exit status 0 and `SUMMARY.json` status `COMPLETE`, with `cohort=16`,
   `complete=16`, `produced_archives=96` and zero failures. Inspect all 48 source
   records and all 96 archive hashes/readbacks. Partial outputs are not complete
   reproduction; failed teachers are retained in the 16-entry census.

The pin authenticates manifest bytes, not authority or truthful selection
provenance. The validator cannot derive or verify historical selection from
unreleased upstream files. A false origin hash remains a false claim even when
its syntax passes; custody verification is a prerequisite, not a new algorithm.

## Separate artificial-only reviewer checks

After permission to execute these tests in an existing suitable environment:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -m unittest -v test_storage
```

These tests create temporary artificial arrays, including the specified shapes,
then serialize them. They access no real model/data files, do not time operations,
and do not run fitting, forward passes, scientific jobs or registry operations.
The full artificial cohort test verifies 16 IDs × 3 model states × 2 encodings,
not a historical measurement. Tests may directly call the storage function to
separate algorithm checks from the exact-version Linux CLI gate. A pass in a
different environment is not historical-byte reproduction. Tests were not run
by the source packager; reviewer evidence must be recorded separately.

## Interpretation and limitations

Whole teacher/C/J parameter and archive-byte ratios use the **entire model**,
not replacement-only budgets. Geometric aggregation, minima/maxima and
smaller/tie/larger counts are retained. A partial run reports its smaller complete
count and is `INCOMPLETE`; its ratios are descriptive of successful rows only.
Compression is fixed once per codec, not tuned. No latency, memory-at-inference,
hardware-general speed or runnable-deployment conclusion follows.

NPY/ZIP content bytes are deterministic for identical tensors in the fixed
NumPy/zlib environment. Environment/manifest/source audit metadata and output
path choices are not claims of byte-identical historical result folders. No
historic result numbers have been recalculated for this source export.
