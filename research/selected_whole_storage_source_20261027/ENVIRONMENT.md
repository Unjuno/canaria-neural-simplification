# Storage-only environment recipe

Historical recorded versions: **Python 3.13.5**, **NumPy 2.3.5**, **zlib 1.3.1**,
little-endian Linux x86_64. This package does not require PyTorch, sklearn, SciPy,
a dataset, a model framework, a GPU or a service account. The `torch.float32`
string in logical hashing is a historical byte-format constant, not an import.

## Provisioning specification (not executed)

Use an independently approved existing Linux environment or separately approved
provisioning. The interpreter must report Python 3.13.5 and both
`zlib.ZLIB_VERSION` and `zlib.ZLIB_RUNTIME_VERSION` 1.3.1. zlib is an interpreter
build/runtime dependency, not a pip requirement. `requirements.lock.txt` pins the
sole third-party Python dependency to `numpy==2.3.5`. Set
`OPENBLAS_NUM_THREADS=1` before starting Python. No install/download command was
run for this source package.

The CLI checks those Python/NumPy/zlib versions, Linux and little-endian byte
order. It retains the 256 MiB address-space limit before NumPy import and a
600-second SIGALRM, one process and no subprocess launches. Use a clean process;
already-loaded numerical libraries or a large host application can make the cap
incompatible. An environment/bounds failure is not permission to raise limits,
change versions, reuse outputs, or retry a scientific run.

`manifest_contract.py` uses only the standard library. It can be reviewed without
NumPy. Artificial tensor tests need NumPy in an authorized existing environment;
their direct function calls intentionally do not alter process resource limits
or assert historical environment identity. Test passes on a different interpreter
are algorithm/portability evidence, not historical-byte validation.

## Known limits of the recipe

The original interpreter build, wheel checksums and complete OS image are not
available in this public package. Exact version pins are necessary provenance,
not proof of identical binary builds. ZIP_STORED depends on identical NumPy NPY
encoding; DEFLATE bytes additionally depend on zlib behavior. Reproduction claims
must compare actual archives with authorized original identities and document the
runtime used, not infer equality from version strings. The CLI records compile
and runtime zlib versions separately; the historical environment record had only
the compile-time zlib version, so the stronger runtime gate is a portability guard.

No environment was installed or changed during packaging. The historical full
model measurement and any separately approved artificial tests must not be
conflated with execution of this source-only candidate.
