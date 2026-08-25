# Project status

**2026-08-25: research consolidation / public-snapshot phase.**

Broad experiment expansion is paused. Canaria has enough confirmatory and negative evidence to preserve a coherent research result. New experiments should now be added only when they close a specific evidential gap required for a public claim or deployment proof-of-concept.

## Current project-level thesis

The central result is **task-conditioned compositional simplification of learned neural computation**:

> implementation-level components that are difficult or expensive to simplify separately can sometimes admit a substantially simpler task-preserving representation when treated as one composed input-output function.

The dynamic extension is:

> **form → transfer → commit → recontract → transfer again**

Intervening task learning after a structural consolidation changes the subsequent optimization geometry. In the current small real-text LM testbed, later compiler fitting becomes easier in normalized functional-error terms, while downstream sensitivity to residual error increases.

## Current evidence frontier

### Confirmed

- historical composition subadditivity in the original confirmatory setting;
- whole-network reductions under declared codecs, including an exact 9,926-byte residual-CNN endpoint;
- training-time staged consolidation (G7);
- function-aligned transfer requirement (G8);
- diminishing returns to transfer fit (G9);
- inheritance + functional refinement (G10);
- autonomous consolidation under a locked non-inferiority protocol (G11);
- staged-vs-direct path effect (G15);
- factorization-without-learning equivalence control (G17);
- deadline-aware controller improvement (G18);
- staged-path replication on `5→4→2` (G19);
- lower normalized next-compiler fit cost after recontracting (G20d);
- higher immediate task sensitivity at matched normalized error (G20e, G22);
- sensitivity-aware immediate-damage prediction (G23–G25);
- horizon-aware future-damage prediction (G26).

### Confirmed negative / boundary results

- Canary is not a necessary local condition for simplification.
- Teacher-forced PPL is not sufficient evidence of autoregressive functional equivalence.
- The tested v23–v25 natural-text post-hoc objectives did not recover rollout-sensitive fidelity.
- A hard task-damage veto (G21) can prevent final contraction and increase compiler cost.
- A single fixed risk cap did not produce a cost/utility Pareto improvement in G27 exploration.
- Unlimited recursive collapse is not supported by the current grammar.

## Reproducibility closure

A portable public runner now reproduces **G7 fresh confirmatory seed 4300** without private `/mnt/data` imports.

On the recorded environment (Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0), the complete reproduced JSON was byte-for-byte identical to the archived historical output after JSON serialization, with matching SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

The progressive compute-matched condition reproduced test PPL **18.932213342799887**, versus early one-shot **19.164090006166735**, late one-shot **19.33549169102473**, and large reference **19.278388330876** for seed 4300.

See:

- `scripts/reproduce/g7_confirmatory/run_seed.py`
- `scripts/reproduce/g7_confirmatory/README.md`
- `results/reproduction/g7_seed4300_report.json`
- `.github/workflows/reproduce-g7.yml`

This is a software/reproducibility result for an already-confirmatory seed, not a new independent scientific replication.

## What remains worth doing

Only conditional closure work remains:

1. **one direct compositional-simplification replication** on a clearly different architecture/task, only if a publication-level generalization/novelty claim requires it;
2. **one minimal runtime-compilation proof-of-concept**, only if systems/deployment claims are to be made.

Everything else should be treated as future work for interested researchers rather than required completion work.

## Current public documentation

- `docs/PUBLIC_SNAPSHOT.md`
- `docs/CORE_DISCOVERY.md`
- `docs/CLAIMS_AND_EVIDENCE.md`
- `docs/TRAINING_TIME_CONSOLIDATION.md`
- `docs/LATE_STAGE_FINDINGS.md`
- `docs/NEGATIVE_RESULTS.md`
- `docs/APPLICATIONS.md`
- `docs/OPEN_QUESTIONS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/PUBLICATION_NOTES.md`
- `docs/TERMINOLOGY.md`
- `docs/FAQ.md`

The repository should be read as an auditable research snapshot, not as a production-ready compression/runtime library.
