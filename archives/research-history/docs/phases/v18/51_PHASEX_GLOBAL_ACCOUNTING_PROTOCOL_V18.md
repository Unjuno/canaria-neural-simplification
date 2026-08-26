# Phase X — Global Complexity Accounting (v18)

Status: preregistered before outcome inspection.

## Objective
Test whether the confirmed 44.5-byte compiled full-span core plus shell repair produces a net reduction in whole-network code length relative to a matched continued-training control, or merely relocates complexity into the shell.

## Seeds
Scan seeds from 2800 upward; include the first 8 with baseline clean accuracy >= 0.95. Exclusion is determined before compilation outcomes.

## Model / intervention
- 8-block residual digits CNN, identical to Phase U.
- Full 8-block core is replaced by the Phase-U 1:4 shared-pattern ternary Conv3 core (nominal 356 bits = 44.5 B).
- Compiled core is frozen.
- stem, b_in, b_out, head are repaired for tau=8 with the same optimizer/data-order convention as the matched control.
- Matched control continues training the full original model for the same tau=8.

## Primary accounting endpoints
All codes are architecture-conditional: tensor shapes/topology shared by the decoder are not charged unless specific to the compiled operator.

1. Fixed FP32 code
   - control: all parameter scalars x 32 bits.
   - compiled: shell scalars x 32 bits + exact core 356 bits.

2. Q8 code
   - symmetric per-parameter-tensor int8 quantization, FP16 scale per tensor.
   - report ideal empirical entropy bits and actual zlib-compressed bytes.
   - compiled core remains charged at exact 356 bits rather than expanded 584-weight q8 representation.
   - q8 functional fidelity must be reported.

3. Rank-99 code
   - each Conv/Linear weight is reshaped out_features x remaining_features.
   - minimum rank retaining >=99% Frobenius energy.
   - factor code is min(dense FP32, FP32 U/V factor scalar count), plus full FP32 bias and 16 bits rank metadata per weight tensor.
   - compiled core remains charged at 356 bits.

4. Shell relocation diagnostics
   - q8 entropy/zlib and rank-99 code separately for stem, b_in, b_out, head.
   - shell repair delta L2 and q8 delta-code relative to the same pre-repair baseline shell.

## Utility
Primary functional criterion: repaired compiled augmentation accuracy / matched-control augmentation accuracy >= 0.95.
Also report clean accuracy and q8 fidelity.

## Decision
Global reduction is supported for an accounting endpoint if the seed-paired mean reduction is >0 and the 95% seed bootstrap CI lower bound is >0, while compiled utility mean is >=0.95 and CI lower bound is >=0.95.

Relocation-only is supported if core savings are offset by shell code increase such that whole-network reduction CI includes <=0.

## H/T/D/C/U
H: whole-network code decreases after confirmed core compilation despite shell repair.
T: first 8 eligible seeds, paired matched control, tau=8, three code families.
D: positive whole-network reduction with lower 95% CI >0 and utility lower CI >=0.95.
C: absolute parameter deletion may dominate fixed code while functional complexity migrates into shell; q8/rank/delta diagnostics test this.
U: code-family dependence, architecture-conditional metadata assumptions, finite seed uncertainty.
