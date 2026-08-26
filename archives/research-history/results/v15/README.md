# v15 results index

Phase I/J/K tested quantizer choice, sparse-support refitting, storage accounting, and FP16 scale metadata.

Protocol/results:
- [`../../docs/phases/v15/35_PHASEI_QUANTIZER_SPARSE_REFIT_PROTOCOL_V15.md`](../../docs/phases/v15/35_PHASEI_QUANTIZER_SPARSE_REFIT_PROTOCOL_V15.md)
- [`../../docs/phases/v15/36_PHASEI_J_K_RESULTS_V15.md`](../../docs/phases/v15/36_PHASEI_J_K_RESULTS_V15.md)

Key conclusions:
- 2-bit performance was strongly quantizer-dependent; per-channel calibration recovered much of the loss seen with one tensor-wide scale;
- 3-bit calibrated dense Conv3 was already close to the FP32 replacement in the tested cohort;
- refitting coefficients on a fixed sparse support materially helped at low K;
- unstructured sparsity paid substantial index overhead, motivating structured sparsity in v16;
- storing per-channel scales as FP16 produced no detected accuracy change in the tested conditions.
