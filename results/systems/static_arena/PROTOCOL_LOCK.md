# Systems S6 — locked protocol

Tracking: Issue #29.

This protocol was committed before native-runtime outcomes.

## Source

- G7 seed-4300 learned compiler blocks.
- Tensor-content SHA256:
  - block0 `78fd7f52ef6f019f6ede72c73b4928b0482d61e7f7914de532ebc084779fce56`
  - block1 `d70098af827694a42e7bb31958cf9959ec3aceef0d432941960c15d0cb5091c8`
- Fixed S4/S5 activation `[2,48,24]`, float32, RNG seed 20260827.
- Fixed S4 PyTorch reference output.

## Native execution

- GCC 14.2.0, C++17, Linux x86_64.
- Statically linked binary (`-static`).
- Minimal chroot execution filesystem: native binary plus raw block/input/reference files only; no Python, PyTorch, NumPy, or SciPy runtime.
- Learned block tensors are packed as little-endian float32 in the fixed state-dict order documented by the exporter.
- One 14,784-byte learned block is loaded at a time.
- Row-streamed causal attention; no full `[B,H,T,T]` score tensor.
- Exact GELU via native `erff`.

## Memory lock

One `alignas(64)` application arena of exactly 65,536 bytes supplies all explicitly managed:

- learned block weights,
- input/output activation,
- LayerNorm activation,
- K/V tensors,
- one causal score row,
- query/context/row scratch,
- final reference tensor used by the verification executable.

A bump allocator records the actual high-water mark. No algorithmic buffer may live outside this arena except small scalar/control locals.

Execution-phase `malloc/calloc/realloc` calls are instrumented with linker wrappers. PASS requires zero post-guard allocation calls.

## PASS rule

All conditions are required:

1. static binary executes inside the minimal chroot;
2. execution filesystem contains no Python/PyTorch/NumPy runtime;
3. both learned blocks execute in order;
4. max absolute output difference vs locked PyTorch reference <= 5e-5;
5. relative L2 output difference <= 1e-5;
6. arena high-water mark <= 65,536 bytes;
7. no full attention-score matrix is created;
8. post-guard heap-allocation violation count = 0;
9. pre-execution raw-file SHA256 and source tensor identity are recorded.

Secondary: static executable bytes, execution latency, compiler stack-usage report.

## Interpretation boundary

A PASS is a native fixed-arena prototype result, not a claim that the complete process or a physical device uses <=64 KiB RAM. libc process state, executable text, OS state and small application stack/control locals are outside the arena. It is also separate from scientific composition-generalization evidence.
