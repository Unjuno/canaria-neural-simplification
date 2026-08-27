# Systems S5 — sub-64-KiB logical working set via row-streamed attention

Status: **PASS** under the locked S5 protocol, with a documented post-outcome status-aggregation bugfix.

## Result

S5 replaces the conventional full causal-attention score tensor with a query-row-at-a-time implementation. The actual learned G7 compiler weights and the fixed S4 activation/reference are unchanged.

- torch imported in execution child: **false**;
- max absolute difference vs PyTorch reference: **7.1526e-7**;
- relative L2 difference: **1.1763e-7**;
- output sum: `333.3709716796875`, exactly matching the reference sum;
- maximum causal score-row payload: **1,536 B**;
- locked explicit managed-tensor working-set upper bound: **54,208 B**;
- capacity target: **65,536 B (64 KiB)**;
- conventional full score tensor alone: **73,728 B**;
- conventional `one block weights + input + full scores` lower bound: **97,728 B**, before other activations;
- median two-block execution in this Python/NumPy prototype: **4.66 ms**.

## Runtime schedule

For each block, the runtime keeps one learned block, `x`, pre-attention LayerNorm output `z`, and full K/V. Q and causal scores are created one query row at a time; softmax is performed in-place on the row score buffer. `x` is updated row-by-row, and `z/K/V` are released before the MLP phase.

The code never creates a `[B,H,T,T]` score tensor.

## Status aggregation bug

The first execution produced all passing measured conditions but top-level `status: FAIL`. The original runner stored the factual condition `full_score_tensor_created: false` directly inside a dictionary passed to `all()`, so that false value incorrectly forced the status to fail.

`POST_OUTCOME_STATUS_BUG.md` records the first-run values before correction. Commit `53390278579dfc1b6fd618cd85524126754ad5a8` changes only the status predicate to the positive pass condition `no_full_score_tensor_created: true`. No algorithm, threshold, input, learned weight, memory accounting, or measured result changed.

## Interpretation

At the level of explicitly managed float32 tensor payload, learned-weight streaming plus row-streamed attention moves this locked two-block compiler invocation below a 64-KiB logical working-set target.

This is a materially stronger small-device design result than weight compression alone because it addresses the activation/scratch bottleneck exposed by S4.

## Boundary

This **does not** mean the Python/NumPy process fits in 64 KiB. Interpreter/runtime code, ndarray metadata, allocator overhead, OS page cache, `.npy` headers, and opaque NumPy/SciPy temporary allocations are excluded. A C/C++/embedded runtime with explicit arena allocation is required for an actual 64-KiB device-memory claim.

Machine-readable result: `RESULT.json`.
