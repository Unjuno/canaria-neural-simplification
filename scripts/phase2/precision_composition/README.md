# Phase 2 — precision × composition

This directory contains the first post-v0.2.0 research phase. The frozen public snapshot on `main` is not retroactively modified by these experiments.

## Experiments

- **2A:** 32/12/8/6/4/3-bit post-training quantization on component-wise vs composed replacements.
- **2B:** 3-bit capacity rescue up to 16,384 replacement parameters.
- **2C:** 3-bit per-matrix vs per-output-channel scale granularity.

Read `docs/phase2/PRECISION_COMPOSITION.md` before interpreting results. Protocol locks and recorded outputs are under `results/phase2/precision_composition/`.

## Portable runners

From this directory:

```bash
python phase2a.py --seed 31000 --out-dir /tmp/phase2a
python phase2b.py --seed 31100 --out /tmp/phase2b_seed31100.json
python phase2c.py --seed 31200 --out /tmp/phase2c_seed31200.json
```

Dependencies are the repository's existing PyTorch / NumPy / scikit-learn stack.

## Evidence status

These are new-phase experiments, not amendments to the frozen v0.2.0 claim set. The public runner refactor is source-reviewed and syntax-checked separately from the historical execution scripts. Phase 2C seed 31200 was rerun through the portable refactor and matched the recorded JSON object exactly; broader byte-identical reproduction should be verified before making that stronger claim for all phases.
