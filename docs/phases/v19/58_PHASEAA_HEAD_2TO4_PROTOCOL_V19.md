# Phase AA — 2:4 structured sparse classifier head (exploratory, v19)

Designed after low-rank <10KB attempts were unstable.

Model: repaired compiled model as v18. Keep exact 356-bit core. For head first linear 512->48, in each consecutive group of four input weights per output row retain the two largest-magnitude weights (2:4 mask). The support is selected without labels. Refit only the retained coefficients by ridge least squares to reproduce the original first-layer preactivation on the first 512 unlabeled training activations. Quantize retained weights with one calibrated signed 4-bit scale per output row; bias FP16. Final head layer and all other shell weights use channelwise 4-bit; biases FP16.

Storage includes retained values, exact 3-bit code per 2-of-4 pattern, FP16 scales/biases, exact core, and 16-bit codec metadata.
Initial cohort: Phase-Z4 seeds 3000-3007 (exploratory). If successful, freeze condition and confirm on new seeds.