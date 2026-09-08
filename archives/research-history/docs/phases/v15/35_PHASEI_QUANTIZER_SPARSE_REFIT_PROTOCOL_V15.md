# Phase I protocol — low-bit quantizer × sparse weight refit (v15)

Status: exploratory follow-up to locked Phase H.

## Fixed before evaluation
- architecture/training: identical residual-8 digits model used in Phase H
- seed rule: starting at 1800, take the first 8 seeds with baseline clean accuracy >= 0.95
- full span: blocks 0..7
- reference: one fitted Conv3 (584 scalar weights including bias)
- no Canary use
- no shell repair in primary tests

## I1 — dense quantizer
Bits: {2,3,4,5,6,8,12,32}
Modes:
1. tensor_max: one symmetric scale for all 584 scalars
2. channel_max: one symmetric scale per output channel (8 scales), weight+bias jointly per output channel
3. channel_calibrated: per-output-channel scale selected from clipping multiplier grid {0.35,0.5,0.65,0.8,0.95,1.10,1.25} to minimize calibration span MSE; stored result is still one scale/channel

Primary readout: held-out augmentation utility and retention relative to the seed's FP32 Conv3.

## I2 — 4-bit sparse count
K = {48,64,80,96,112,128,144,160,176,192,224,256}
Masks:
- global_mag: globally largest |weight| scalars
- balanced_mag: approximately K/8 scalars per output channel
Coefficient states:
- inherited: retained fitted Conv3 values
- refit: fixed sparse support, coefficients re-solved by ridge least squares on calibration span pairs
All sparse cores are then quantized with 4-bit channel_calibrated quantization.

## I3 — short repair sensitivity
Only the best sparse-refit family selected without task utility peeking (balanced_mag + refit by construction) is evaluated with full-shell tau=2 at K={64,96,128,160,192}. Matched continued-training controls use the same data order.

## Decision discipline
This is exploratory. Report seed-cluster bootstrap CIs. Do not relabel as confirmatory.