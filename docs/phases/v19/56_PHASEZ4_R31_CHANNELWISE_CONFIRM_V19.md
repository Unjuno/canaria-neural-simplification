# Phase Z4 — Independent confirmation of <10 KB whole model

Condition frozen from Phase Z3: rank31 head factorization; all non-core weights signed calibrated 4-bit with one FP16 scale per output row/channel; all biases FP16; exact 356-bit core; 16-bit architecture metadata; no extra repair.

Cohort: starting seed 3000, first 8 baseline-clean>=0.95 seeds.

Primary endpoints: packed bytes; combined fidelity vs compiled FP32; compression fidelity vs dense compiled channelwise-q4; quantized utility vs matched control channelwise-q4.

Decision: report each separately; full <10KB whole-model PASS requires packed<10000 and bootstrap lower 95% CI >=0.95 for combined fidelity and quantized utility. Head-compression-specific PASS requires lower CI >=0.95 for compression fidelity vs dense compiled q4.