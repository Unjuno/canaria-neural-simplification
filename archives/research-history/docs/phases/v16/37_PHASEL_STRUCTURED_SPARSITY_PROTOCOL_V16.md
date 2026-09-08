# Phase L protocol — structured sparsity × low-bit storage (v16)

Status: exploratory follow-up to locked Phase I/J/K.

## Fixed before evaluation
- architecture/training: residual-8 digits model identical to Phase I
- seed rule: starting at 1900, take the first 8 seeds with baseline clean accuracy >= 0.95
- full span: blocks 0..7
- reference replacement: fitted Conv3, 8 input x 8 output x 3 x 3 + 8 bias = 584 scalars
- no Canary use
- primary tests use no shell repair
- calibration uses the same first 192 training samples as Phase I
- all retained coefficients are ridge-refit on the fixed structured support, then quantized by per-output-channel calibrated symmetric quantization
- bits: {3,4}; scale metadata: 8 FP16 scales = 128 bits

## Structured support families
1. kernel_block: each of 64 input-output 3x3 kernels is an indivisible group. Keep R in {8,12,16,20,24,32,40,48,56,64}; choose by fitted-Conv3 group L2 norm. Biases always retained. Mask metadata = 64 bits.
2. input_channel: keep C_in in {1,...,8}; each retained input channel keeps all output channels and 3x3 positions. Choose by group L2 norm. Biases retained. Mask metadata = 8 bits.
3. spatial_offset: keep P in {1,...,9} shared 3x3 offsets across all input/output channels. Choose by group L2 norm. Biases retained. Mask metadata = 9 bits.
4. semistructured_nm: for each output row, split the 72 non-bias weights into 18 consecutive groups of 4. Keep N={1,2,3,4} largest-magnitude entries per group. Biases retained. Pattern metadata per group = ceil(log2(C(4,N))) bits, except N=4 uses 0 bits.

## Storage accounting
nominal_bits = retained_weight_count * weight_bits + support_metadata_bits + 8*16 scale bits.
Biases are included among retained_weight_count and quantized with the output channel weights.

Dense controls: calibrated dense 2/3/4-bit Conv3 using the same implementation; dense storage includes 8 FP16 scales.

## Primary readouts
- augmentation utility relative to baseline model
- retention relative to same-seed FP32 Conv3 reference
- PASS95 relative to baseline
- span relative MSE
- nominal storage bytes including support metadata and scales

## Repair follow-up
After the no-repair table is complete, take Pareto structured conditions with retention >= 0.98 and nominal bytes < dense-3-bit storage and evaluate tau=2 full-shell repair with matched continued-training control. No other structured condition receives repair.

## Decision discipline
Exploratory. Report seed-paired/bootstrap intervals. Do not relabel as confirmatory.