# v13 results index

Phase G separated **precision per weight** from **number of stored per-function coefficients**.

Public protocol: [`../../docs/phases/v13/32_PHASEG_FLOAT_BUDGET_PROTOCOL_V13.md`](../../docs/phases/v13/32_PHASEG_FLOAT_BUDGET_PROTOCOL_V13.md)

Key conclusions carried forward:
- keeping all 584 Conv3 scalars at 12-bit was effectively indistinguishable from FP32 in the tested cohort;
- 8-bit was also close; 4-bit showed measurable but limited degradation;
- reducing the function to only 4 or 12 per-function coefficients was not sufficient;
- weight precision and weight-count are therefore distinct bottlenecks and must not be conflated.

The later v14/v15 phases refine these thresholds with finer bit/count sweeps and improved quantizers.