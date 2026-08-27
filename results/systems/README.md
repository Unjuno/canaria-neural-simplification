# Systems results

Read `docs/SYSTEMS_RUNTIME.md` first for the current evidence chain and claim boundaries.

Current stages:

- `streaming_runtime/` — S1 synthetic compact streaming mechanism;
- `learned_streaming/` — S2 actually learned G7 compiler streaming;
- `memory_ceiling/` — S3 explicit Linux memory-ceiling feasibility boundary;
- `numpy_streaming/` — S4 framework-independent learned compiler kernel;
- `row_streamed_attention/` — S5 sub-64-KiB logical managed-tensor schedule;
- `static_arena/` — S6 statically linked C++ fixed-arena prototype.

Each stage preserves its own protocol/result/provenance. Later stages strengthen the runtime mechanism but do not retroactively change earlier protocols or turn systems evidence into scientific composition-generalization evidence.
