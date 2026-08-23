# Reproducibility guide

## Environment

The historical runtime dependencies inferred from scripts are:

- Python 3.x
- PyTorch
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib

Early historical experiments did **not** preserve an exact package lockfile. Reproduce qualitative/aggregate behavior first; do not expect bitwise identity from the oldest scripts.

## Statistical unit

Repeated span/composition events within one trained network are not independent. Confirmatory inference uses the **training seed/network** as the cluster unit. Seed-cluster bootstrap or leave-one-seed-out evaluation is preferred to naive event-level confidence intervals.

## Blindness rule used in Phase A

1. Train eligible baseline networks.
2. Evaluate simplification candidates **without computing Canary**.
3. Save the complete Stage-1 table.
4. Hash-lock Stage 1 with SHA256.
5. Only then compute Canary and join the tables.

See `results/phaseA_v11/STAGE1_LOCK.json` and the locked CSVs.

## Eligibility

For the decisive residual-8 experiments, baseline clean accuracy had to meet the predefined eligibility floor before Canary/simplification outcomes were measured. Failed baseline seeds were not converted into positive/negative simplification observations.

## Utility controls

Repair experiments compare a compiled model against a **matched continued-training control** receiving the same additional training budget. Absolute accuracy alone is not sufficient, because both the compiled and uncompiled networks can improve with extra epochs.

## Precision terminology

Custom 2/3/4/12-bit experiments are fixed-grid research quantizers unless a hardware datatype is explicitly named. Do not describe the 4-bit experiments as hardware FP4 unless the script actually uses that format.

## Storage terminology

- `core bytes` refers only to the compiled replacement module.
- `whole-network bytes` includes all model components encoded by the specified codec.
- entropy/ideal code lengths are not necessarily real file sizes.
- the 9,926-byte v19 codec is a real serialized whole-model size and was roundtrip-tested.

## Recommended reproduction order

1. Run repository audit.
2. Reproduce one baseline/residual-8 training seed.
3. Reproduce the locked Phase-A blind-map logic on a small seed subset.
4. Reproduce Phase X global accounting.
5. Reproduce the v19 exact 9,926-byte codec.

Historical scripts are intentionally preserved; for new work, create a new phase script rather than silently modifying an old experiment.
