# Residual-MLP core-discovery reproduction

This directory contains the clean reproduction path for the strongest direct component-wise-versus-composed experiment currently on `main`.

## Evidence class

The original fresh-seed cohort `1200–1207` is **confirmatory scientific evidence** under `results/core_discovery_digits/PROTOCOL_LOCK.json`.

Re-running those already-observed seeds is **reproduction evidence**, not a second independent confirmatory cohort.

## Pinned environment

The current pre-announcement reproduction target is:

- Python `3.11.x` (CI reference: `3.11.16`);
- NumPy `2.4.6`;
- PyTorch `2.13.0`;
- scikit-learn `1.9.0`.

Install CPU PyTorch explicitly so a CPU-only reproduction does not pull the default CUDA dependency stack:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts/reproduce/core_discovery_digits/requirements-pinned-py311.txt
python -m pip install -e .
```

The second `pip install` keeps `torch==2.13.0` in the dependency record; an already-installed CPU build satisfying that public version should not be replaced by the default CUDA build.

## One seed

```bash
python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 \
  --out /tmp/canaria_seed1200.json
```

Recorded primary endpoint:

```text
component-wise minimum passing budget: 3072
composed minimum passing budget:       1536
```

## Full fresh cohort reproduction

Run:

```bash
python scripts/reproduce/core_discovery_digits/verify_confirmatory.py \
  --out results/reproduction/core_discovery_digits_pinned_env_report.json
```

The verifier:

1. runs the existing evidence-producing runner for seeds `1200–1207`;
2. compares each selected component-wise/composed budget with `results/core_discovery_digits/confirm_summary.json`;
3. independently reconstructs mean budgets, mean paired log2 ratio, geometric budget ratio, and the protocol-locked 20,000-resample bootstrap interval;
4. compares the selected-endpoint test-accuracy mean within a small numerical tolerance;
5. records Python/package versions and the current git commit when available.

A PASS means the already-observed headline cohort is reproducible under this pinned modern environment. It does **not** add new scientific seeds or establish external validity.

## Integrity boundary

Do not edit `run_confirmatory.py` merely to make a reproduction pass. If a portability change becomes necessary, add a new versioned runner and explain the difference. A dependency-driven reproduction failure is itself useful evidence and should be recorded rather than hidden.