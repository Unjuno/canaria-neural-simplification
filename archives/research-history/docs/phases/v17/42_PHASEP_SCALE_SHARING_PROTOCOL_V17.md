# Phase P — 1:4 2-bit scale-sharing compression (v17)

## Purpose
Test whether the independently validated 1:4, 2-bit, 90-byte representation can be reduced below 90 bytes by sharing FP16 quantization scales across output channels.

## Independent cohort
First 8 baseline-eligible seeds (clean >= 0.95) in ascending order starting at 2300.

## Fixed representation
1:4 semi-structured support, fixed from FP32 fitted Conv3 by absolute magnitude within each group of 4 weights. Support coefficients are ridge-refit. Quantization is signed 2-bit with calibrated shared scale(s), and every stored scale is actually rounded to FP16 before reconstruction/evaluation.

## Conditions
Number of output-channel scale groups S in {1,2,4,8}. Since there are 8 output channels:
- S=1: one shared scale, nominal 76 B
- S=2: nominal 78 B
- S=4: nominal 82 B
- S=8: per-output scale, nominal 90 B reference

Pattern metadata remains 2 bits per 1:4 group (288 bits total). Stored values include 144 selected kernel weights plus 8 biases at 2 bits each.

## Evaluation
No-repair utility/retention and matched-control full-shell repair at tau=2.

## Primary decision
A sub-90-byte condition is stable if tau=2 mean utility has seed-bootstrap lower 95% CI >= 0.95. No condition is changed after hash lock.