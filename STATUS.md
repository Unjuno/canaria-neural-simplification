# Project status

**Current mode: public handoff / release preparation. Broad experiment expansion is stopped.**

The repository now has enough evidence to be useful as a public research object without requiring a complete theory or exhaustive architecture sweep. New experiments should be limited to small checks that materially improve reproducibility, correct an identified defect, or remove a release blocker.

## Core result retained

The project-level thesis is **task-conditioned compositional simplification of learned neural computation**:

> implementation-level components that are difficult or expensive to simplify separately can sometimes admit a substantially smaller task-preserving representation when treated as one composed input-output function.

The core effect has direct locked replications in:

- a Small Vision Transformer; and
- a residual MLP with exactly matched replacement-parameter budgets.

In the residual-MLP replication, the composed condition selected a lower minimum passing budget in **8/8 fresh seeds**; geometric composed/component-wise budget ratio **0.4823×**.

## Minimal public entry point

Run one direct residual-MLP seed:

```bash
python -m pip install numpy torch scikit-learn
python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 \
  --out /tmp/canaria_seed1200.json
```

Recorded seed 1200: component-wise `3072` parameters versus composed `1536`.

See [`QUICKSTART.md`](QUICKSTART.md).

## Precision follow-up

Post-snapshot Phase 2 extended the question to quantization.

Current supported boundary:

- 4-bit composed coded-size advantage: retained under the locked residual-MLP experiment;
- naive 3-bit per-matrix PTQ: not rescued by simply increasing weight count;
- row-wise scales: can rescue 3-bit PTQ;
- short correctly implemented activation-domain QAT repair: can also make coarse per-matrix 3-bit viable in this model family;
- a lower QAT repair sample complexity for the composed condition: **not confirmed** in the 24-seed confirmatory test.

## Critical correction

Phase 2E is **INVALIDATED_IMPLEMENTATION_BUG**.

Its repair code used raw digit inputs `Xt` where the replacement was defined on the internal activation domain `ta[0]`. The tensors happened to have the same width, so the error was silent.

The invalid artifact is retained and marked, not erased. Follow-up Phase 2L changed only the intended repair input and restored composed passing from `0/8` to `8/8` under the controlled rerun.

See [`docs/phase2/README.md`](docs/phase2/README.md) and [`results/phase2/precision_composition/CORRECTION_STATUS.json`](results/phase2/precision_composition/CORRECTION_STATUS.json).

## What is not established

The repository does not claim:

- universal simplification of arbitrary neural networks;
- codec-independent Kolmogorov/MDL reduction;
- large-pretrained-LLM validity;
- task- or span-universal compositional subadditivity;
- guaranteed FLOP, wall-clock, energy, RAM, VRAM, or GPU improvements;
- a confirmed compositional advantage in quantization-repair sample complexity.

## Stopping rule

Before public release, continue only work in these categories:

1. repository cleanup that makes the core pattern easier to notice or reproduce;
2. correction/provenance work for already-run experiments;
3. one-seed or similarly small smoke tests needed to verify a public runner;
4. release metadata, tags, CI, and repository protection.

Do **not** start a new broad experimental family merely to make the repository feel complete. Unresolved architecture, theory, hardware, and scale questions should remain open for follow-up by other researchers.

## Release separation

The original v0.2.0 scientific snapshot remains the frozen baseline until its tag/release boundary is completed. Post-snapshot precision work is staged separately and should not be back-projected into the v0.2.0 claim registry without an explicit version transition.

For the detailed historical record, use the documents under `docs/`, `results/`, and `archives/`; the root README is intentionally kept short.