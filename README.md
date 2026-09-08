# Canaria

**Task-conditioned neural simplification — research code, evidence, and limitations.**

[日本語](README.ja.md) · [Reproduce](QUICKSTART.md) · [Evidence](docs/CLAIMS_AND_EVIDENCE.md) · [Research index](docs/RESEARCH_INDEX.md)

Canaria asks whether a trained neural-network span can be replaced by a smaller computation when it is fitted as a composed input–output function rather than simplified one implementation block at a time.

**This is a research preview, not a universal compression method or a production inference library.** `main` provides the baseline experiment and a bounded evidence registry. Newer experiments are indexed separately at immutable commits; linking them does not promote them into the headline claim.

> **Publication check: strict full-cohort numerical reproduction is unresolved.** Two complete repeat executions retained the directional 8/8 pattern but not all archived endpoints/statistics. See [the disclosed discrepancy](docs/REPRODUCTION_DISCREPANCY.md). Do not infer announcement readiness from a one-seed smoke test.

## Start with the recorded result and its reproduction status

In the recorded **residual-MLP / sklearn digits / first-two-block** experiment, the composed replacement needed fewer learned replacement parameters in all eight model seeds. “Minimum” means the smallest passing point on the tested budget grid, not a mathematical minimum.

| Recorded result | Component-wise | Composed |
|---|---:|---:|
| Seed 1200: minimum passing replacement parameters | 3,072 | 1,536 |
| Seeds 1200–1207: mean selected replacement parameters | 3,584 | 1,728 |

[Locked protocol](results/core_discovery_digits/PROTOCOL_LOCK.json) · [Complete recorded summary](results/core_discovery_digits/confirm_summary.json) · [Scope and interpretation](docs/CORE_DISCOVERY_REPLICATION_DIGITS.md)

Validation selects candidate budgets; candidate test metrics are evaluated at selected endpoints. The teacher and a prespecified mechanistic control also have test metrics; test is excluded from budget selection. These are **replacement parameters**, not total model parameters, bytes, FLOPs, latency, or memory. The eight models share one dataset split; they are not eight independent dataset replications.

## Run it

From a checkout of this repository, using **Python 3.11 on CPU**:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts/reproduce/core_discovery_digits/requirements-pinned-py311.txt
python scripts/reproduce/core_discovery_digits/run_confirmatory.py --seed 1200 --out outputs/seed1200.json
```

[Quickstart](QUICKSTART.md) includes the full eight-seed verifier, expected results, environment checks, and failure reporting. A rerun of these already-observed seeds is **reproduction**, not additional confirmatory evidence.

## What is — and is not — supported

The headline is a bounded empirical comparison under one declared architecture, span, replacement family, and passing rule. See the [claim registry](docs/CLAIMS_AND_EVIDENCE.md) for supporting baseline work and explicit exclusions.

We do **not** claim universal or LLM-scale compression, a general minimum-dimension predictor, or general runtime, RAM, VRAM, or energy savings. Recent head-derived correction experiments reduce a **correction subspace**, not the 4,096-parameter final model; they still compute full teacher residuals during calibration.

**Correction history is part of the evidence:** Phase 2E is `INVALIDATED_IMPLEMENTATION_BUG` and `DO_NOT_USE_FOR_INFERENCE`. Phase 2O did not establish a repair-sample advantage. [Corrections and negative results](docs/NEGATIVE_RESULTS.md)

## Explore without confusing evidence classes

| Entry | What it contains |
|---|---|
| [Claims and evidence](docs/CLAIMS_AND_EVIDENCE.md) | The headline, supporting baseline, caveats, and exclusions |
| [Research index](docs/RESEARCH_INDEX.md) | Recursive composition, systems, Gaussian-shift studies, and C75E–C77E; pinned source/evidence links |
| [Reproducibility](docs/REPRODUCIBILITY.md) | Checks, statistical units, environment limits, and reporting failures |
| [Status and readiness](STATUS.md) | Research-preview boundary and the current announcement gate |
| [Archives](archives/README.md) | Preserved legacy protocols/results, invalidation history, and path migration |

The frozen tag `v0.2.0-public-snapshot` remains historical provenance, not a certificate of current readiness. No research protocol or outcome is rewritten by this reorganization.

## Contribute and cite

See [CONTRIBUTING.md](CONTRIBUTING.md). Reproduction failures and well-controlled counterexamples are useful contributions. Cite the **exact commit and experiment artifacts**, not an unversioned claim. [CITATION.cff](CITATION.cff)

Original code and documentation: [Apache-2.0](LICENSE). Third-party data and dependencies retain their own licenses.
