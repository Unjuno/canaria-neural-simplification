# Portable G7 confirmatory reproduction

This directory provides a self-contained reproduction of **fresh confirmatory seed 4300** from the G7 training-time consolidation experiment.

It exists to remove the historical `/mnt/data` import assumptions from the evidence script without changing the scientific computation.

## What is reproduced

The runner compares the same six conditions used in G7:

- large reference (`depth 4`, MLP 48);
- small from start (`depth 2`, MLP 24);
- terminal post-hoc consolidation;
- late one-shot consolidation;
- early one-shot consolidation;
- progressive compute-matched `4→3→2` consolidation.

The data are deterministic character-LM windows generated from the dataset-description text files shipped with scikit-learn. No private dataset files are required.

## Install

A minimal current environment is:

```bash
python -m pip install torch numpy scikit-learn
```

The exact environment used for the public reproduction on 2026-08-25 was:

- Python 3.13.5
- PyTorch 2.10.0+cpu
- NumPy 2.3.5
- scikit-learn 1.8.0

Historical runs did not preserve a full lockfile, so other library versions may reproduce the qualitative endpoint without byte-identical floating-point output.

## Run

From the repository root:

```bash
python scripts/reproduce/g7_confirmatory/run_seed.py \
  --seed 4300 \
  --out g7_seed_4300.json
```

For the environment listed above, the complete JSON output has SHA256:

```text
68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028
```

The headline test perplexities are:

| condition | test PPL |
|---|---:|
| large reference | 19.278388330876 |
| small from start | 20.49408599251734 |
| terminal post-hoc | 19.30211076363844 |
| late one-shot | 19.33549169102473 |
| early one-shot | 19.164090006166735 |
| progressive compute-matched | **18.932213342799887** |

See `results/reproduction/g7_seed4300_report.json` for the recorded reproduction result.

## Integrity note

The public runner changes only path/import portability relative to the archived G7 confirmatory code. Model definitions, dataset generation, seeds, minibatch order, optimizer settings, learning-rate schedule, compiler budgets, replacement sequence, and metrics are unchanged.

The exact match is a **reproducibility result**, not a new independent scientific replication: seed 4300 was already part of the original fresh confirmatory cohort.
