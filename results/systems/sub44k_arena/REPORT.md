# Systems S7 — sub-44-KiB native execution arena

Status: **PASS**.

S7 follows the S6 static C++ fixed-arena runtime. It keeps the same learned G7 seed-4300 two-block compiler, raw payload, fixed activation, native kernel, static linking, chroot isolation, row-streamed attention, and heap-allocation guard. The memory schedule alone is made more deployment-oriented:

1. the verification-only PyTorch reference is not resident in the execution arena; the native runtime writes `output.bin`, which is checked outside the chroot;
2. the full pre-attention LayerNorm tensor `z[B,T,D]` is not retained. Each row is normalized into one 24-float scratch buffer to build K/V, and the row is recomputed later for Q.

## Locked gate

The protocol fixed an arena capacity of **45,056 B (44 KiB)** before outcomes. Numerical gates remained max absolute difference <= `5e-5` and relative L2 <= `1e-5`. No post-guard heap allocation, full `z`, or full `[B,H,T,T]` score matrix was allowed.

## Result

- measured arena high-water: **43,808 B / 45,056 B**;
- post-guard heap-allocation violations: **0**;
- max absolute difference versus the locked PyTorch reference: **9.5367431640625e-7**;
- relative L2 difference: **1.2255255243296686e-7**;
- output SHA256: `7021dfd394d0820fb3bdc65e90b00eb83e5191cfb7371e42a7ddf822da02372f`;
- five independent chroot executions produced the same output SHA256 and all passed the external verifier;
- median two-block native execution time across those five runs: about **0.474 ms** in this x86-64 environment;
- static executable: 684,488 B; compiler-reported `main` stack usage: 2,416 B.

S6 measured a 62,272-B arena high-water because it also held the 9,216-B verification reference and retained full `z`. S7 reduces measured arena high-water by **18,464 B**, to about `0.7035x` the S6 value. This comparison intentionally combines both production-oriented changes and should not be described as the isolated effect of recomputation alone.

## Interpretation boundary

The supported systems statement is narrow: this learned compact compiler can execute in the native prototype with **43,808 B of explicitly managed application arena** for the locked `[2,48,24]` activation while preserving the reference output within the preregistered tolerance.

This is **not** a claim that the total process uses 44 KiB, that a physical 44-KiB MCU has been demonstrated, or that arbitrary neural models admit this compact form. Executable text, libc/process state, operating-system state, and small stack/control locals are outside the arena. Scientific composition claims remain separate.
