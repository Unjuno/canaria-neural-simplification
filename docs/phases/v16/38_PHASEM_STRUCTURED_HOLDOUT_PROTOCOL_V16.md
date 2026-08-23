# Phase M protocol — independent structured-sparsity holdout (v16)

Status: independent holdout validation after exploratory Phase L.

## Fixed before holdout evaluation
- seed rule: starting at 2000, first 8 seeds with baseline clean >= 0.95
- same residual-8 model, full span 0..7, Conv3 reference, 192 calibration samples
- no Canary
- support selection and ridge refit identical to Phase L
- per-output-channel calibrated quantization, scales stored FP16

## Primary holdout conditions
- kernel_block R=24, 3-bit (108 B nominal)
- kernel_block R=32, 3-bit (135 B nominal)
- semistructured 2:4, 3-bit (181 B nominal)
- spatial_offset P=5, 4-bit (181.125 B nominal)
Controls: dense 2-bit, dense 3-bit, dense 4-bit, FP32 Conv3.

## Fine kernel-block sweep
R={20,24,28,32,36,40}; bits={2,3}.

## Readouts
No-repair: utility vs baseline, retention vs same-seed FP32 Conv3, PASS95.
Repair: primary four structured conditions receive tau=2 full-shell repair; matched continued-training control uses identical repair data/order.

## Holdout decision
Report all 8 eligible seeds. A structured condition is considered holdout-stable if mean retention >=0.98 and its seed-bootstrap 95% lower bound >=0.95. Stable repair if PASS95 >= 6/8 and bootstrap lower bound of repair utility >=0.95.