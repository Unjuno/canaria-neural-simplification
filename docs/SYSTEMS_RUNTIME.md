# Canaria systems/runtime evidence

This page is the current index for runtime and constrained-memory experiments. These experiments are **systems evidence**, not scientific evidence that compositional simplification generalizes across arbitrary tasks or architectures.

## Evidence chain

| Stage | Question | Result | Strongest supported statement | Important boundary |
|---|---|---|---|---|
| S1 | Can an already-compact representation avoid full dense materialization? | PASS | Synthetic compact factors can be streamed with much lower resident payload/RSS than full dense materialization. | Synthetic exact low-rank operator stack. |
| S2 | Does the mechanism work on an actually learned Canaria compiler? | PASS | G7 seed-4300 learned compiler can be loaded one block at a time with exact model outputs; learned-payload amplification shows large RSS reduction. | 4096 logical chunks are a memory-measurement harness, not a trained 4096-block model. |
| S3 | Is there a memory ceiling where full retention fails but streaming completes? | PASS | Under locked Linux `RLIMIT_AS` +64 MiB headroom, full retention failed at 2181/4096 chunks while streaming completed 4096/4096 with exact checksum. | Linux constrained-process harness, not a physical-device deployment. |
| S4 | Can the learned compiler run outside PyTorch? | PASS | Raw learned tensors plus an independent NumPy/SciPy kernel reproduce PyTorch output to ~1e-7 relative L2. | Python/NumPy runtime still dominates real process memory. |
| S5 | Can attention scratch be streamed too? | PASS | Row-streamed attention removes the full score matrix and reduces the locked logical managed-tensor bound to 54,208 B. | Logical tensor accounting only; Python/runtime overhead excluded. |
| S6 | Can the learned compiler run in a native fixed arena? | PASS | Statically linked C++ runtime executes the learned two-block compiler in a minimal chroot with 62,272-B fixed-arena high-water and zero guarded heap allocations. | S6 includes a verification reference inside the arena; not total-process RAM. |
| S7 | Can the native execution schedule fit below 44 KiB of application arena? | PASS | Removing verification-only reference storage and recomputing LayerNorm rows reduced measured native arena high-water to **43,808 B** with the same locked output tolerance and zero guarded heap allocations. | Application arena only; not a physical 44-KiB-device or total-process RAM claim. |

## Current systems claim boundary

A communication-safe summary is:

> An actually learned Canaria compact compiler can be stored as raw learned data and executed block-by-block without materializing the full learned payload at once. In the current G7 prototype, the execution path has been reproduced outside PyTorch and in a statically linked C++ fixed-arena runtime. With a production-oriented recomputation schedule, the locked native prototype uses 43,808 bytes of explicitly managed application arena for the fixed `[2,48,24]` activation. A Linux memory-ceiling harness also shows a regime where full retention fails while streaming completes.

Do **not** replace that with any of the following without new evidence:

- “Canaria runs arbitrary neural networks in 44 KiB.”
- “A 44-KiB MCU has been demonstrated.”
- “The whole native process uses 43,808 bytes.”
- “All models admit the same compact representation.”
- “Systems results prove the scientific composition claim.”

## Where to read the evidence

- S1: `results/systems/streaming_runtime/`
- S2: `results/systems/learned_streaming/`
- S3: `results/systems/memory_ceiling/`
- S4: `results/systems/numpy_streaming/`
- S5: `results/systems/row_streamed_attention/`
- S6: `results/systems/static_arena/`
- S7: `results/systems/sub44k_arena/`

Runtime sources are under `scripts/systems/`.

## Next systems step

The next meaningful step is hardware-level validation: port the S7 schedule to a specific constrained target or embedded simulator/toolchain and measure **total** RAM/flash/latency on that target. Until that exists, S7 is the strongest current native-runtime evidence boundary.