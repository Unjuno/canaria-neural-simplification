# Quickstart — one direct composition experiment

The shortest useful experiment in this repository is the residual-MLP component-wise-versus-composed replacement test on `sklearn.datasets.load_digits`.

## Convenience run

For an exploratory local run using your current environment:

```bash
python -m pip install numpy torch scikit-learn
python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 \
  --out /tmp/canaria_seed1200.json
```

This is convenient, but it is **not** the preferred evidence-reproduction path because package versions are unconstrained.

## Pinned reproduction environment

For reproducibility, use Python 3.11 and the recorded dependency set:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts/reproduce/core_discovery_digits/requirements-pinned-py311.txt
python -m pip install -e .
```

Then run one seed:

```bash
python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 \
  --out /tmp/canaria_seed1200.json
```

For the full headline-cohort reproduction, follow [`scripts/reproduce/core_discovery_digits/README.md`](scripts/reproduce/core_discovery_digits/README.md).

## What the runner does

The runner trains a four-block residual MLP, then replaces the first two residual blocks in two ways while matching learned replacement-parameter count at every budget:

```text
component-wise:
  block 0 replacement + block 1 replacement

composed:
  one replacement trained directly on block1(block0(x))
```

The endpoint rule uses validation span NMSE and validation accuracy. Test accuracy is evaluated only after the minimum passing budget has been selected.

## Recorded result for seed 1200

```text
component-wise minimum passing budget: 3072
composed minimum passing budget:       1536
log2(composed / component-wise):       -1.0
```

The complete locked 8-seed confirmatory summary is in [`results/core_discovery_digits/confirm_summary.json`](results/core_discovery_digits/confirm_summary.json).

Across seeds `1200–1207`:

```text
component-wise mean minimum passing budget: 3584
composed mean minimum passing budget:       1728
composed lower:                             8 / 8 seeds
geometric budget ratio:                     0.4823×
```

## Why this is the preferred first experiment

It tests the central phenomenon directly and does not depend on the later training-time controller, quantization, runtime, or historical G-number experiments. If the pattern is interesting, the next two documents are:

- [`docs/CORE_DISCOVERY_REPLICATION_DIGITS.md`](docs/CORE_DISCOVERY_REPLICATION_DIGITS.md)
- [`docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md`](docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md)

For the broader evidence boundary, see [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md). For current announcement-readiness status, see [`docs/ANNOUNCEMENT_READINESS.md`](docs/ANNOUNCEMENT_READINESS.md).