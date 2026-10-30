# Selected whole-model latency — source-only derivative

## Separate fresh-replication handoff

[FRESH_INPUTS.md](FRESH_INPUTS.md) specifies the new explicit `fresh-v1` input
and release mode, immutable selection review gate, exact existing saved-reference
mapping and metadata-only preparation helper. It does not replace historical v1
or provide inputs, approval, execution or historical reproduction. The original
three historical provenance pins and frozen CSV pin remain unchanged.

The inspection/execution statements below describe the original source-package
preparation. This fresh revision has a separate, explicitly artificial-only test
proof; no real scientific payload or benchmark is executed.

**Inspectability package, not a completed reproduction or a timing release.**
No candidate module was imported or executed, no supplied test was run, and no
model, array or dataset was loaded while this package was prepared. Only static
text/AST/JSON/hash checks were performed. No weights, data, model archives, raw
measurements, logs, private manifests or operational registry are bundled.

This narrow package closes the source dependency chain for the previously fixed
16-teacher / selected C/J whole-model latency procedure. It does not train, select,
acquire data, change scientific thresholds or provide a general benchmark framework.
The consumed historical attempt remains separate and immutable. Its timing is a
conditional observation on one shared CPU host, **not a portable or universal
speed claim**. No new independent scientific evidence is created here.

## Contents and status

- [PROTOCOL.md](PROTOCOL.md): unchanged graph/timing/summary rules and explicit
  historical-versus-portable lifecycle distinction.
- [ARTIFACTS.md](ARTIFACTS.md), [INPUT_SCHEMA.json](INPUT_SCHEMA.json),
  [RELEASE_SCHEMA.json](RELEASE_SCHEMA.json), [OUTPUT_SCHEMA.json](OUTPUT_SCHEMA.json):
  missing payloads, custody and future private-attempt contracts.
- [ENVIRONMENT.md](ENVIRONMENT.md): exact historical runtime/profile and limits.
- [TRANSFORMATION.md](TRANSFORMATION.md), [SOURCE_MAP.json](SOURCE_MAP.json),
  [STATIC_VALIDATION.json](STATIC_VALIDATION.json): provenance and limits of checks.
- `model_definitions.py` + `deployment.py`: exact graph definitions, strict loader
  and fixed validation correctness prerequisite; numerical imports only in worker.
- `timing_core.py`: frozen order/timer/summary and evidence writer.
- `latency_contract.py` + `storage_manifest_contract.py`: local hash-bound handoff.
- `benchmark.py` + `supervise.py`: sole supervised, one-attempt private execution path.
- `test_controls.py`: fake clocks/callables and artificial metadata only.

## Inspection-first review

A reviewer should first inspect every file in the exact allowlist, then inspect
`test_controls.py`. Only after separate permission for synthetic tests, the
reviewer may run from this directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v test_controls
```

That command is **not recorded as executed** by this package. It neither imports
the deployment module nor opens scientific archives. The full-worker test replaces
all numerical/data/model operations and the timing clock with fakes. Passing these
controls would establish control behavior, not model correctness or timing validity.

## Future scientific execution is not authorized here

There is deliberately no valid release, input-manifest instance or runnable sample
model/data. A separately authorized custodian must supply every exact input and
reviewed private mapping described in ARTIFACTS.md. A fresh reviewed checkout,
explicit execution approval, independent GO and no-parallel-benchmark attestation
must precede any future `supervise.py` invocation. It requires explicit `--out`,
`--release`, `--manifest`, `--manifest-sha256`, `--input-root`, and `--control-root`.
There is no preparation inference mode, dataset fallback, resume, automatic retry,
new selection or command that reopens the historical attempt.

All generated attempt outputs are private by default. In particular
`PRIVATE_ENV_*` and `PRIVATE_FAILURE_TRACEBACK.txt` contain host/process/path data
and must never be included in a blanket public directory upload. This publication
contains source only; no generated artifact or future publication is authorized.
