# Systems S1 — streaming compact runtime

Status: **PASS** under the locked exploratory systems protocol in `PROTOCOL_LOCK.json`.

This experiment tests runtime representation/materialization only. It is not scientific evidence that arbitrary trained neural spans admit the compact representation.

## Setup

The same stack of 32 residual linear operators is represented either as expanded dense matrices or as exact low-rank factors:

```text
compact: x <- x + 0.05 * (x @ U) @ V
expanded: W = U @ V; x <- x + 0.05 * x @ W
```

Locked scale: float32, `d=1024`, rank `r=64`, 32 blocks, batch 8, RNG seed `20260827`.

Three fresh-process execution modes were measured:

1. `dense_resident` — all dense blocks resident before inference;
2. `dense_streaming` — one dense block memory-mapped at a time;
3. `factor_streaming` — one `(U,V)` pair memory-mapped and executed directly, never constructing `W`.

## Result

| mode | peak RSS delta | peak model payload | median inference |
| --- | ---: | ---: | ---: |
| dense resident | 129.33 MiB | 128.0 MiB | 41.09 ms |
| dense streaming | 5.27 MiB | 4.0 MiB | 49.64 ms |
| factor streaming | 0.91 MiB | 0.5 MiB | 8.91 ms |

Factor streaming used `0.00390625×` the peak model payload of dense-resident execution and `0.00701×` the measured RSS delta in this process/runtime.

Serialized representation size was 128.00 MiB for the expanded dense blocks and 16.01 MiB for the factor representation (`0.1251×`).

The locked 32 MiB logical model-payload capacity witness was satisfied: the full dense-resident representation requires 128 MiB of model payload, while factor streaming needs only 0.5 MiB of model payload at one time.

## Output agreement

- dense streaming vs dense resident max absolute difference: `0.0`;
- factor streaming vs dense resident max absolute difference: `1.4305e-6`;
- factor streaming relative L2 difference: `2.6999e-7`.

All locked numerical-equivalence checks passed.

## Interpretation

This is direct evidence for a **runtime mechanism**: when a function is already available in a compact factor representation, a runtime can execute it block-by-block without expanding the full dense representation, materially reducing simultaneous resident model bytes. In this synthetic low-rank case, the native compact operator also required less computation and was faster than dense execution.

This result does **not** establish that a real trained model can always be converted to such a representation, does not demonstrate deployment on a particular MCU/mobile/edge device, and does not strengthen the scientific compositional-simplification claim.

The next useful systems experiment is to connect an actually learned Canaria replacement to this streaming execution path and then measure the same memory endpoints.

## Provenance

- issue: #19
- protocol was committed before execution in commit `71f1aaa6e435f4d7b805270a6b61d43fc3867f03`;
- protocol blob: `3b46e38c85fa30fc48762455598cac114c3dc110`;
- runner blob: `e44621cf960a427a2b72d63e6eb73c09918e7243`;
- local runner Git blob hash was verified equal to the committed runner blob before execution;
- runtime: Python 3.13.5, NumPy 2.3.5, psutil 7.2.2, Linux x86_64;
- probe subprocesses used one BLAS thread.

Machine-readable outcome: `RESULT.json`.
