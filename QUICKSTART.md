# Quickstart: reproduce the current R87R2 baseline

The current reviewed direct baseline is **R87R2**, not the historical seeds 1200–1207 archive. R87R2 is a fixed residual-MLP / sklearn-digits / first-two-block experiment under the scoped `portable_v2_sqrt64` CPU numerical profile. Repeating the fixed 16 seeds is a **technical reproduction**, not 16 new scientific observations and not recovery of the historical archive.

See [the R87R2 protocol](results/reproduction/r87r2_fresh_sqrt_profile/PROTOCOL.json) and [the historical discrepancy](docs/REPRODUCTION_DISCREPANCY.md) before interpreting a rerun.

## 1. Clean checkout and locked R87R2 environment

Use an x86-64 CPU environment where PyTorch reports effective AVX2 capability. The reviewed hosted environment is Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0, and SciPy 1.17.0. This is a scoped tested profile, not a general CPU/GPU portability guarantee.

```bash
git clone https://github.com/Unjuno/canaria-neural-simplification.git
cd canaria-neural-simplification
git rev-parse HEAD

python3.13 -m venv .venv-r87r2
source .venv-r87r2/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts/reproduction_diagnostics/requirements-r87r2-py313.txt
python -m pip check
```

On Windows, activate with `.venv-r87r2\Scripts\activate`. The experiment itself has been validated only within the declared tested host/profile boundary; Windows, macOS, GPU, non-AVX2, timing, and resource parity are not claimed.

## 2. Run the fixed evaluator tests and complete 16-seed cohort

```bash
python scripts/reproduction_diagnostics/r87r2_confirm.py --mode selftest
python scripts/reproduction_diagnostics/r87r2_confirm.py \
  --mode suite --workers 2 --out outputs/r87r2-clean
```

The suite first runs the locked seed-1202 implementation bridge, then seeds `871200`–`871215`. Missing or malformed seeds are failures; do not substitute seeds or widen gates after seeing results.

The expected locked decision is `R87R2_CONFIRMATORY_PASS`. The reviewed evidence contains 16/16 eligible seeds, geometric composed/component-wise selected-budget ratio `0.5316098878`, and selected test-accuracy noninferiority under the preregistered 2 percentage-point margin. These are model-seed results on one reused dataset split, not independent-dataset replication.

## 3. Require exact agreement with the reviewed primary evidence

After the suite completes:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('outputs/r87r2-clean')
seeds = range(871200, 871216)
expected_rows = json.loads(Path(
    'results/reproduction/r87r2_fresh_sqrt_profile/FRESH_ROWS.json'
).read_text())
expected_decision = json.loads(Path(
    'results/reproduction/r87r2_fresh_sqrt_profile/DECISION.json'
).read_text())
actual_rows = [
    json.loads((root / f'seed_{s}' / 'outcome.json').read_text())
    for s in seeds
]
actual_decision = json.loads((root / 'DECISION.json').read_text())

assert expected_rows['primary_host'] == 'a'
assert actual_rows == expected_rows['rows']
assert actual_decision == expected_decision
print('R87R2 exact technical reproduction: PASS')
PY
```

The publication candidate runs the same clean-checkout check in `.github/workflows/publication-headline-reproduction.yml`. Exact equality here is a technical reproducibility result for the fixed cohort/profile. It does not increase the scientific sample size or prove cross-platform universality.

## 4. Audit the publication evidence without retraining

```bash
python tools/audit_publication.py
python tools/audit_publication_candidate.py
python tools/audit_repo.py
```

`audit_publication_candidate.py` independently recomputes the saved R87R2 and California Housing Phase4 aggregate decisions from persisted raw rows and verifies reviewed Git-blob identities. Repository CI is code/evidence auditing, not external peer review or independent external reproduction.

## 5. Historical 1200–1207 archive: separate reproduction question

The historical digits cohort remains preserved, but current pinned-environment reruns did not reproduce every archived endpoint/statistic exactly. Issue #87 therefore remains open. **Do not use R87R2 to rewrite or close that historical discrepancy.**

For the historical runner, use a separate Python 3.11 environment:

```bash
python3.11 -m venv .venv-historical
source .venv-historical/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts/reproduce/core_discovery_digits/requirements-pinned-py311.txt
python -m pip install -e .

python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 --out outputs/historical-seed1200.json
python scripts/reproduce/core_discovery_digits/verify_confirmatory.py \
  --out outputs/historical-core-cohort-report.json \
  --raw-dir outputs/historical-core-cohort-raw
```

Seed 1200 historically records selected replacement budgets 3072 component-wise versus 1536 composed. The archived 1200–1207 summary records means 3584 versus 1728 and composed lower in 8/8. Those historical numbers are provenance records; the full modern strict verifier is expected to disclose the known mismatch rather than loosen tolerances.

[Original historical protocol](results/core_discovery_digits/PROTOCOL_LOCK.json) · [Recorded historical summary](results/core_discovery_digits/confirm_summary.json) · [Reproduction reports](results/reproduction/README.md)

The separate strict historical gate remains:

```bash
python tools/audit_publication.py --require-reproduction
```

It is intentionally unresolved while Issue #87 remains open.

## 6. Inspect supporting and research-only evidence

Use the [research index](docs/RESEARCH_INDEX.md) and [claims/evidence registry](docs/CLAIMS_AND_EVIDENCE.md). California Housing Phase4 is bounded regression support; SmallViT is narrow supporting architecture-family evidence. Research PRs, negative results, invalidated evidence, and systems prototypes keep their own scopes and are not promoted merely because an artifact says `PASS`.

## Report a mismatch

Retain the complete output directory and process logs. Report the exact commit, command, Python/package versions, CPU/OS, PyTorch CPU capability, thread settings, expected/observed endpoints, and full error. Do not replace a seed, alter the fixed profile, change thresholds, or widen tolerances after observing a mismatch. A mismatch is evidence to investigate; it is neither automatically a refutation nor a successful reproduction.
