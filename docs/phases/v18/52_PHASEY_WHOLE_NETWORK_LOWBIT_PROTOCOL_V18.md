# Phase Y — Whole-network low-bit executable accounting (v18)

Preregistered after Phase-X global-accounting decision and before Phase-Y outcomes.

## Seeds/model
Reuse deterministic seeds 2800–2807 and the exact Phase-X training/intervention: matched tau=8 control and Phase-U 44.5 B compiled core with tau=8 shell repair.

## Quantization
Quantize all non-core parameter tensors of both models independently at b in {3,4,6,8} bits using symmetric signed fixed-grid quantization. Per tensor, choose scale from a fixed alpha grid by weight MSE only (no validation accuracy), store scale as FP16. The compiled ternary core remains in its exact 356-bit code.

## Endpoints
- quantized augmentation fidelity relative to each model's FP32 state.
- compiled-vs-control utility after both are quantized at the same b.
- architecture-conditional packed bytes: b bits per scalar + 16-bit scale per parameter tensor; compiled adds exact 356-bit core.

## Decision
A bit width is called whole-model functionally viable if compiled-vs-control mean utility lower 95% seed-bootstrap CI >=0.95 and compiled self-fidelity lower CI >=0.98.