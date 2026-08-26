# Reproducibility guide

Canaria distinguishes **confirmatory science**, **exploration**, **valid negative/boundary evidence**, **invalidated evidence**, and **reproduction**. A rerun of an already-observed seed validates portability; it does not add a new scientific seed.

## Repository integrity

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -e .
python -m unittest discover -s tests -v
python tools/audit_repo.py
```

An audit PASS checks repository structure and selected semantic invariants. It is not an announcement-readiness certificate and does not imply every historical experiment is bitwise reproducible.

## Headline direct experiment — residual MLP on digits

The original fresh seeds `1200–1207` are confirmatory evidence under `results/core_discovery_digits/PROTOCOL_LOCK.json`.

For a convenience one-seed run:

```bash
python -m pip install numpy torch scikit-learn
python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 \
  --out /tmp/canaria_seed1200.json
```

Recorded seed 1200 selects component-wise budget `3072` and composed budget `1536`. The convenience command resolves current packages and is therefore not the preferred evidence-reproduction path.

### Pinned modern reproduction environment

Use Python 3.11 and:

```bash
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts/reproduce/core_discovery_digits/requirements-pinned-py311.txt
python -m pip install -e .
```

Pinned top-level versions are NumPy 2.4.6, PyTorch 2.13.0 CPU, and scikit-learn 1.9.0. The CPU PyTorch index avoids downloading a CUDA stack for this CPU-only reproduction.

For the complete already-observed cohort:

```bash
python scripts/reproduce/core_discovery_digits/verify_confirmatory.py \
  --out results/reproduction/core_discovery_digits_pinned_env_report.json
```

The verifier reruns `1200–1207`, checks all selected budgets, and reconstructs the aggregate primary statistics and protocol-locked bootstrap interval. A PASS is reproduction of existing confirmatory evidence, not a second independent replication.

The residual-MLP runner fits replacements on training activations, selects the minimum passing budget using validation data, then evaluates test utility for the selected endpoint.

## SmallViT isolation note

The SmallViT locked selection rule excludes test accuracy, but its runner records test metrics for every candidate. Therefore say **“test was not a selection variable”**, not “test remained operationally hidden until after selection.” Future confirmatory runners should delay test evaluation until selection is frozen.

## G7 portable reproduction

`scripts/reproduce/g7_confirmatory/` contains the recorded portable seed-4300 reproduction. Its output exactly matched the archived result with SHA256 `68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`. This is portability evidence for an already-confirmatory seed.

## Phase 2 boundary

Phase 2A–C have checked-in protocol/result files and portable runners. Phase 2E is `INVALIDATED_IMPLEMENTATION_BUG`: repair used raw `Xt` instead of internal activation `ta[0]`. Its `0/8` outcome is preserved as correction history and must not support inference. Later 2D–2O correction provenance is identified by SHA256 `1a339be12d7644de534ac77a712307c49ee0c3d9acb28c8a3532883edca3dab7`; not all later raw per-seed artifacts are checked into Git.

## Test and statistical discipline

For new confirmatory work: train on training/calibration data, select on declared validation/holdout criteria, freeze the endpoint, then evaluate final test outcomes. Unless another hierarchy is preregistered, treat independently initialized training seed/model as the inferential unit.

Keep parameter count, optimizer-update proxies, nominal bits, serialized bytes, FLOPs, wall-clock latency, energy, RAM, and VRAM distinct. A reduction in one does not establish reduction in the others.

## Current pre-announcement gate

The active gate is Issue #13 and `ANNOUNCEMENT_READINESS.md`, not the historical Issue #9/v0.2.0 checklist. Before broad announcement, require:

1. pinned full-cohort residual-MLP reproduction PASS;
2. explicit inclusion/exclusion decision for candidate external-validity evidence;
3. final integrated claim/communication review;
4. unit tests, `tools/audit_repo.py`, and GitHub `repository-audit` PASS on the final candidate commit.
