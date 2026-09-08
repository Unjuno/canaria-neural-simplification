# Quickstart: reproduce the baseline, then inspect research

> **Known mismatch:** the full modern pinned-environment rerun did not match all archived endpoints. The strict verifier is expected to report a discrepancy on affected stacks; do not loosen it. [Details](docs/REPRODUCTION_DISCREPANCY.md). Seed1200 alone is not a cohort validation.

## 1. Get a clean checkout

```bash
git clone https://github.com/Unjuno/canaria-neural-simplification.git
cd canaria-neural-simplification
git rev-parse HEAD
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts/reproduce/core_discovery_digits/requirements-pinned-py311.txt
python -m pip install -e .
```

The documented reproduction target is **Python 3.11, PyTorch 2.13.0 CPU, NumPy 2.4.6, scikit-learn 1.9.0**. These numerical dependencies are pinned; the exact OS, CPU, Python patch version, and complete installed packages are recorded by the validation workflow. This is not a hermetic cross-platform guarantee. On Windows activate the environment with `.venv\Scripts\activate`; Windows/macOS/GPU parity is not claimed by a Linux CPU run.

The core runner uses the digits data shipped with scikit-learn: no private files, credentials, pretrained weights, or external data service are needed. Installation requires network access. Reproduction should not be confused with a hardware benchmark.

## 2. Reproduce one already-observed seed

```bash
python scripts/reproduce/core_discovery_digits/run_confirmatory.py --seed 1200 --out outputs/seed1200.json
```

Expected selected replacement budgets: component-wise **3072**, composed **1536**. These exclude the unchanged stem/head and are not complete-model parameter counts. The minimum is over the fixed tested grid. The runner trains the teacher and candidates; this is not only JSON readback.

## 3. Reproduce the complete existing cohort

```bash
python scripts/reproduce/core_discovery_digits/verify_confirmatory.py --out outputs/core-cohort-report.json --raw-dir outputs/core-cohort-raw
```

The verifier runs all **1200–1207** seeds using the unchanged scientific runner, retains each raw output and process log, checks the selected budgets exactly, and checks numerical statistics with documented tolerances. A missing/failed seed is a failure, not permission to replace it. The recorded means are component-wise **3584** and composed **1728**, with composed lower in **8/8**. Reproduction does not increase the original confirmatory sample size.

[Original protocol](results/core_discovery_digits/PROTOCOL_LOCK.json) · [Recorded summary](results/core_discovery_digits/confirm_summary.json) · [Reproduction reports](results/reproduction/README.md)

## 4. Audit without training

```bash
python -m unittest discover -s tests -v
python tools/audit_repo.py
python tools/audit_publication.py
```

The last two commands check files, links, correction status, scope policy, and preservation of baseline science. They do not claim that every historical experiment was retrained or independently reviewed.

## 5. Inspect the newer research separately

Use the [research index](docs/RESEARCH_INDEX.md). Each entry identifies a fixed commit, protocol, result, code and review surface. Do not copy a research result into the baseline merely because its file is named `PASS`.

For the latest numerical evidence audit, use a **separate checkout** so this baseline remains untouched:

```bash
git clone https://github.com/Unjuno/canaria-neural-simplification.git canaria-research-audit
cd canaria-research-audit
git checkout --detach fcef9b538f3af5ba5ebe318e17b630f1e8e6d3e8
```

Follow the workflow and runner identified in the index; the audit and training paths have different dependencies. Do not run old one-shot workflows without reading their permissions and output paths.

## Report a mismatch

Retain the report and raw outputs. Include commit, command, Python/package versions, CPU/OS, thread settings, expected/observed endpoints and full error message in a reproduction issue. Do not change the protocol, replace an inconvenient seed, or widen a tolerance after seeing a mismatch. A mismatch is evidence to investigate; not automatically a refutation or a successful reproduction.
