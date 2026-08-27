# Systems S6 — static C++ fixed-arena runtime

Status: **PASS**.

S6 takes the learned G7 seed-4300 compact compiler out of Python entirely. The two learned compiler blocks were verified by deterministic tensor-content SHA256, packed into raw little-endian float32 files, and executed by a statically linked C++17 binary inside a minimal `chroot` containing only the binary and four raw data/reference files.

## Result

Primary chroot run:

- max absolute output difference vs the locked S4 PyTorch reference: `9.53674316e-7`;
- relative L2 difference: `1.22552552433e-7`;
- output sum: `333.370965779` vs reference `333.3709716796875`;
- application arena high-water mark: **62,272 B / 65,536 B**;
- post-guard heap allocation violations: **0**;
- full `[B,H,T,T]` score matrix: **not materialized**;
- two learned block sums matched the locked source values within the predeclared `1e-4` check;
- native two-block kernel time in the primary run: about `0.650 ms`.

Five additional chroot repetitions all passed with identical numerical outputs. Median measured native kernel time across those five runs was about `0.538 ms` in this CPU environment.

## Isolation

The execution root contained exactly:

- `bin/canaria_s6`;
- `data/block0.bin`;
- `data/block1.bin`;
- `data/input.bin`;
- `data/reference.bin`.

No Python, PyTorch, NumPy or SciPy runtime was present. The executable was statically linked; `ldd` reported `not a dynamic executable`.

The static executable was 684,488 bytes. This is a prototype binary-size observation, not an optimized embedded flash result.

## Actual arena layout

The fixed arena high-water mark was 62,272 bytes. It includes the learned block buffer, activation buffers, row-streamed attention scratch and the final reference tensor used only for verification.

The executable therefore enforces, rather than merely estimates, that its explicitly managed tensor data fit inside the 64-KiB arena.

A linker-wrapper negative control intentionally called `malloc` after enabling the guard and produced one violation, confirming that the zero-allocation result in the S6 execution path was observable by the instrumentation.

## What this establishes

S6 supports the systems statement that an actually learned Canaria compiler can be represented as raw learned data plus a small native execution kernel and can execute block-by-block with a fixed 64-KiB application arena for explicitly managed weights, activations and scratch.

## What this does not establish

This is **not** a claim that the complete Linux process uses 64 KiB RAM. libc process state, executable text, operating-system state and small stack/control locals are outside the arena. It is also not a deployment result on a particular MCU, phone, SBC or accelerator, and it is not evidence for scientific composition generalization.

For machine-readable provenance and exact hashes, see `RESULT.json` and `RAW_MANIFEST.json`.
