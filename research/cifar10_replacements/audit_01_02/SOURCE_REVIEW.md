# Pinned implementation and historical-evidence limits

Reviewed source commit: `7a3046da9033d2b87b634d791585f480925edccc`; numerical source file bytes are identical in result commit `bb30713cca06519a000251a26668309b1a4ef82b` and match the approved plans and protocol. Protocol freeze `9e48fbb` precedes approval commits `0229f2a`, `2e7656a` and result commit `bb30713`. Exact approved plan Git blobs match local bytes; approval commit timestamps precede recorded launch timestamps.

## Frozen teacher and no repair

- `core.py:51–81`: teacher construction applies `requires_grad_(False).eval()`; `train()` forces evaluation mode; feature extraction and installed prediction use `no_grad`; prefix is blocks `[0:4]`, teacher targets blocks4 and5, and suffix starts block6. Every installed prediction uses the original teacher suffix/classifier.
- `core.py:46–48` and `runtime.py:63–72,99–119`: Adam receives only each replacement arm's parameters, and only those optimizers step. Checkpoints contain 2 Comp or4 Sep convolution tensors per arm, exactly5120 entries; saved optimizer state has exactly those parameter IDs, matching shapes, finite Adam moments and step1000 for every parameter. There is no saved teacher optimizer in a replacement checkpoint.
- `core.py:84–93`: Sep-local uses detached teacher-forced middle endpoint with the SUM of two independently averaged MSE losses; Sep-joint and Comp-joint use the common endpoint. Source matches preregistered objective/dataflow; this audit performed zero optimizer steps.
- `runtime.py:230–258`: calibration-only image accessor is used in fitting, exact scheduled IDs are checked on each step, initial checkpoint precedes the fit loop, and fixed update1000 precedes final validation. Labels are first requested in the validation generator. No test evaluator appears in the entry/data/fit path inspected.

## Independent recomputation

`audit.py` imports no production experiment module. It uses a separately written functional PyTorch teacher/replacement forward and NumPy float64 statistics. It loads checkpoints with `weights_only=True`, permits only the five training tar members, independently recreates splits/normalization and reconstructs seeded initialization and schedules. Teacher state hashes before/after validation are identical; functional batchnorm runs with `training=False`.

## What this can and cannot establish

All saved artifact and numerical checks pass. It does not replay the training trajectory or prove each past batch/gradient from a per-update event log (none retained), and the fit saved no final teacher-state snapshot to compare with its starting teacher. Therefore historical freezing is supported by pinned source and optimizer isolation, not independently proved by before/after teacher tensors. The independent evaluation uses the unchanged original teacher and reproduces the reported outputs. Hashes provide integrity/binding, not authentication or evidence against all hypothetical unlogged activity. This is validation-only evidence for two of three intended seeds; no generalization or statistical noninferiority claim follows.
