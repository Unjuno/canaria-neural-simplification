# Systems S4 — framework-independent streaming kernel for a learned Canaria compiler

Status: **PASS** under the locked S4 protocol.

S4 exports the two actually learned G7 seed-4300 compiler blocks to raw NumPy tensor files and executes them block-by-block in a separate NumPy/SciPy runtime process that never imports PyTorch.

## Result

- source learned tensor hashes: verified against S3 locks;
- one-block learned tensor payload: **14,784 B**;
- two-block learned tensor payload: **29,568 B**;
- raw `.npy` serialized bytes for both blocks: **32,640 B**;
- NumPy execution child imported torch: **false**;
- PyTorch reference output sum: `333.3709716796875`;
- NumPy streaming output sum: `333.3709716796875`;
- max absolute output difference: **7.1526e-7**;
- relative L2 output difference: **1.1291e-7**;
- median NumPy end-to-end two-block execution: **3.04 ms** in this CPU environment.

All locked equivalence checks passed.

## What this establishes

The learned compiler is not tied to PyTorch module materialization. Its actual learned tensors can be exported to a simple raw format and executed by an independent runtime implementing LayerNorm, causal multi-head attention, residual connections, exact GELU, and the MLP.

This strengthens the runtime-format interpretation from S2/S3: a compact learned operator can be represented as data plus a small execution kernel and loaded one block at a time.

## New limiting factor

The model payload is now small enough that **activation/scratch memory becomes the next systems bottleneck**. For the locked `[2,48,24]` activation, the input itself is 9,216 B, while materializing the conventional attention score tensor requires 73,728 B. Thus further edge-oriented work should optimize attention/activation scratch rather than only shrinking weights.

## Boundaries

The input is a fixed synthetic activation used only for kernel equivalence. S4 is not task-utility evidence, does not demonstrate a specific MCU/phone/SBC deployment, and does not support the scientific composition claim.

Machine-readable result: `RESULT.json`.
