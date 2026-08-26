# Phase Z3 — Channelwise 4-bit low-rank head (exploratory, v19)
Designed after Phase-Z rank32 per-tensor-q4 primary failed.

Goal: test whether the failure is quantizer granularity rather than low-rank head approximation.
Condition: rank31 first-head factorization, no additional repair. Every stored weight tensor uses signed calibrated 4-bit quantization with one FP16 scale per output channel/row. Every bias is stored directly in FP16. Exact 356-bit core is unchanged. Low-rank operator metadata = 16 bits.
Comparison uses the same quantizer for matched control and dense compiled model.
Cohort: the same eight Phase-Z seeds 2900–2907; this is exploratory, not independent confirmation.
Success signal: nominal packed whole model <10,000 B and bootstrap lower 95% CI >=0.95 for combined fidelity and quantized utility.