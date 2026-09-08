# Phase AB-confirm — Independent confirmation of 9,926 B model
Frozen condition from Phase AB: dense Conv3 full-span core quantized to calibrated signed 4-bit (296 B), tau=8 full-shell repair, head first linear magnitude 2:4 per output row with original surviving values, retained head weights channelwise calibrated 4-bit, all other shell weights channelwise 4-bit, all biases FP16. Pattern index is exact 3 bits per 2-of-4 group. Total nominal storage 9,926 B including core and metadata.
Cohort: starting seed 3100, first 8 baseline-clean>=0.95 seeds.
Primary: lower 95% seed-bootstrap CI >=0.95 for combined fidelity and quantized utility; packed bytes <10,000. Secondary: compression fidelity vs dense compiled q4.
