# Phase X/Y — Global complexity accounting and whole-network low-bit results (v18)

## Phase X: whole-network complexity accounting
Eight preregistered eligible seeds 2800–2807 were evaluated with a matched tau=8 continued-training control and the confirmed Phase-U 44.5 B compiled core plus tau=8 shell repair.

### Utility
- compiled / matched-control augmentation utility: mean 0.98843
- seed-bootstrap 95% CI: [0.97159, 1.00502]
- individual U>=0.95: 7/8

### Whole-network reduction
Architecture-conditional code, including the exact 356-bit core code:

- Fixed FP32: control 142,824 B; compiled 105,492.5 B; reduction 26.1381% (identical across seeds).
- Q8 ideal empirical-entropy code: mean reduction 27.6536%, 95% CI [27.0292%, 28.3195%].
- Q8 actual zlib code: control mean 31,297.25 B; compiled mean 22,291.25 B; reduction 28.7863%, 95% CI [28.2695%, 29.3688%].
- Rank-99 factor-or-dense code: reduction 26.1381%. Under this simple factor code no shell/control tensor gained enough low-rank advantage to beat dense storage, so this endpoint collapses to the fixed-code result.

Q8 fidelity was high:
- control q8 / control FP32: mean 0.99780, CI [0.99562, 0.99997]
- compiled q8 / compiled FP32: mean 1.00001, CI [0.99814, 1.00225]

### Shell relocation diagnostics
The shell does show a small increase in ideal q8 description length after compilation/repair:
- total shell q8 ideal change: +1,529.6 bits on average, 95% CI [+19.9, +3,174.7] bits.
- head accounts for almost all of this mean increase (+1,512.4 bits).
- actual zlib shell change: +160.6 B mean, CI [-4.9, +337.3] B (not decisive).

However, the removed block code is much larger:
- removed blocks q8-ideal change: -67,411.8 bits mean.
- removed blocks zlib change: -9,008 B mean.

Therefore positive shell q8-ideal growth offsets only 2.59% of removed-block q8 savings on average (95% CI 0.65–4.79%). For zlib it offsets 2.09% on average (CI 0.56–3.87%, with zero-increase seeds counted as zero offset).

Repair-delta diagnostics also do not show a large extra shell-information burden:
- compiled/control shell relative-delta-L2 ratio mean 1.0449, CI [0.9935, 1.1041].
- compiled/control shell-delta q8-zlib ratio mean 1.0043, CI [0.9826, 1.0322].

Decision: global reduction PASS for fixed FP32, q8 ideal, q8 actual zlib, and the preregistered rank-99-or-dense code, while mean utility CI lower bound remains above 0.95. A small shell redistribution signal exists in q8 ideal code, but it is far too small to offset the deleted-core code under these codecs.

## Phase Y: whole-network low-bit executable code
The same deterministic eight seeds and trained states were regenerated. All non-core tensors in both control and compiled models were quantized at the same bit width using fixed-grid symmetric quantization with per-tensor FP16 scales chosen by weight-MSE only. The compiled core remained exact at 356 bits.

### 3 bit
- compiled packed size: 9,950.25 B
- matched control: 13,473.75 B
- reduction: 26.1508%
- compiled self-fidelity: 0.93687, CI [0.91870, 0.95429]
- compiled/control quantized utility: 0.97898, CI [0.95523, 1.00383]
- preregistered viability: FAIL because compiled self-fidelity lower CI <0.98.

### 4 bit
- compiled packed size: 13,245.5 B
- matched control: 17,937 B
- reduction: 26.1554%
- compiled self-fidelity: 0.98541, CI [0.98012, 0.98992]
- compiled/control quantized utility: 0.98124, CI [0.96655, 0.99574]
- preregistered viability: PASS.

### 6 bit
- compiled packed size: 19,836 B
- matched control: 26,863.5 B
- reduction: 26.1600%
- compiled self-fidelity: 0.99778, CI [0.99324, 1.00264]
- quantized utility: 0.99361, CI [0.97119, 1.01321]
- PASS.

### 8 bit
- compiled packed size: 26,426.5 B
- matched control: 35,790 B
- reduction: 26.1623%
- compiled self-fidelity: 0.99852, CI [0.99662, 1.00112]
- quantized utility: 0.98802, CI [0.97131, 1.00458]
- PASS.

## Interpretation bounded by the experiment
Under architecture-conditional fixed, q8-entropy/zlib, and simple rank-factor code families, the confirmed local/core simplification is not merely cancelled by a more complex shell: whole-network code remains about 26–29% smaller while matched-control utility is retained. There is measurable redistribution into the shell under ideal q8 entropy, concentrated in the classifier head, but it accounts for only a few percent of the core savings.

This does not prove a codec-independent minimum description length theorem. It does rule out the strongest form of the relocation-only explanation for the tested architecture/task/code families.
