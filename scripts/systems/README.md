# Systems runtime sources

See `docs/SYSTEMS_RUNTIME.md` for the evidence map and interpretation boundaries.

- `streaming_runtime.py` — S1 synthetic dense vs streaming runtime mechanism.
- `learned_streaming.py` — S2 learned G7 compiler packaging/equivalence/RSS harness.
- `memory_ceiling.py` and `memory_ceiling_tensor_identity.py` — S3 constrained-memory harness; the latter is the pre-outcome identity amendment runner.
- `export_learned_blocks_numpy.py` + `numpy_stream_runtime.py` — S4 raw NumPy export and independent execution kernel.
- `row_streamed_attention.py` — S5 row-streamed causal attention schedule.
- `export_static_arena.py` + `static_arena_runtime.cpp` — S6 deterministic raw packing and statically linked fixed-arena native runtime.
- `sub44k_arena_runtime.cpp` + `verify_sub44k_output.py` — S7 native recomputation schedule below 44 KiB application arena plus external reference verification.

One-shot execution scaffolds should not be added here. Durable protocol, source, result, and audit artifacts belong in the repository; temporary workflow machinery belongs only in producing history.
