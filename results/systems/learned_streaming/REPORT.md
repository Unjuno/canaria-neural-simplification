# Systems S2 — streaming an actually learned Canaria compiler

Status: **PASS** under the locked protocol committed before outcomes at `65517be663de26243809ffe4c2a9398eaac71b4c`.

## What changed from S1

S1 used a synthetic exact-low-rank operator stack. S2 uses the **actually learned G7 seed 4300 progressive `4→3→2` Canaria compiler**. The source compact model was reconstructed from the committed reproduction code; its test PPL was `18.93221334`, matching the existing G7 runtime record.

## S2A — original learned model

The compact model was packaged as a resident shell plus two learned compiler-block chunks. A streamed runtime reused a single `CausalBlock` module, loading block 0, executing it, then loading block 1 and executing it. Both compiler blocks were never required as simultaneously resident live model parameters.

- full compiler tensor payload: **29,568 bytes**
- streamed one-block compiler payload: **14,784 bytes** (`0.5×`)
- max absolute logit difference: **0.0**
- relative L2 logit difference: **0.0**
- test PPL: **18.93221184** in both modes
- token accuracy: **0.22273763** in both modes

All locked S2A equivalence checks passed.

The fresh-process RSS deltas for the original tiny model were 24.73 MB (full) and 24.13 MB (streamed). These are explicitly secondary: the actual compiler is only ~29 kB, so framework/process memory dominates. The single-probe latency numbers are also not promoted as a stable speed claim.

## S2B — learned-payload RSS amplification harness

To measure resident-memory behavior above framework noise, the two **already learned** compiler block states were alternated over 4,096 logical chunks. This is only a payload amplification harness, not a trained 4,096-block model.

- logical learned tensor payload processed: **60,555,264 bytes**
- full-resident peak RSS delta: **112,107,520 bytes**
- streaming peak RSS delta: **1,548,288 bytes**
- streaming / full RSS-delta ratio: **0.01381×**
- checksum: identical (`234473.0256500244`)

The locked S2B gate required streaming RSS delta `< 0.20×` full resident. The observed ratio was `0.01381×`; PASS.

Streaming paid an I/O/materialization cost in this harness (19.50 s versus 10.17 s for retaining all chunks). This is the expected memory-versus-load-time tradeoff and should remain visible.

## What S2 establishes

An actually learned Canaria compiler can be serialized in chunks and executed block-by-block with **exact output equivalence** to the ordinary fully materialized compact compiler. When the same learned payload is scaled only for memory measurement, one-chunk-at-a-time loading keeps peak RSS far below retaining all chunks.

## What S2 does not establish

- no claim that the 4,096-chunk harness is a trained model;
- no claim of universal latency improvement;
- no claim that arbitrary neural networks admit a Canaria replacement;
- no deployment result on a particular MCU, phone, SBC, GPU, or accelerator;
- no strengthening of the scientific compositional-generalization claim.

The next systems experiment should impose an actual process/address-space memory ceiling and test the **feasibility boundary**: a full-resident representation should fail to initialize under the ceiling while the streamed learned-payload runtime completes.

## Provenance

- Issue #21
- protocol blob `e98af89686b146d3e17e031ac32770a1e395dc32`
- S2 runner blob `17fb712bd30c2dded1671d47f5cfe10a13c348f5`
- G7 runner blob `61d8f2c4acce2d9e21daae7680be250ed21a27a9`
- runtime PoC blob `5911bbfbcd9d26d36d95426147c5fedf79ee2c2c`
- Python 3.13.5 / PyTorch 2.10.0+cpu / NumPy 2.3.5 / scikit-learn 1.8.0 / psutil 7.2.2 / one torch thread

Generated artifact SHA256:

- `compact_full.pt`: `77e86d56abc5f0662dbc8d7509a593d82137e158e533297635f90c7c1d1de44a`
- `shell.pt`: `778c3c18cf18d3ba576107d9ed0cc7901e6884759eae909953c2034b8fa45b99`
- `compiler_block0.pt`: `abb2ac470084721948a3521075f1a7f18d08ea61611a3f8a379f8b77d6719daa`
- `compiler_block1.pt`: `7bd667e433e93d9b6e18484bf7d668c95dd5c317c139480a47c1edd20e3c35ca`
- `manifest.json`: `c89e51f19f9ae4a0e660e3fb8c9f97803c8a133fe4bb11f262b7f6afe3e8e8d8`
