# Project status

**2026-08-25: research consolidation / public-snapshot phase.**

Broad experiment expansion is paused. Canaria now has a representative exact public reproduction path and a bounded runtime-materialization proof of concept. New experiments should be added only when they support a deliberately stronger public claim.

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

## Reproducibility closure — complete

A portable public runner reproduces **G7 fresh confirmatory seed 4300** without private `/mnt/data` imports.

On the recorded environment (Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0), the complete reproduced JSON exactly matched the archived confirmatory output with SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

See `scripts/reproduce/g7_confirmatory/` and `results/reproduction/g7_seed4300_report.json`.

This is a software/reproducibility result for an already-confirmatory seed, not a new independent scientific replication.

## Runtime-materialization PoC — complete at small-model scope

A minimal CPU-only end-to-end systems PoC now serializes and materializes the G7 seed-4300 large model and progressive compact model as `state_dict + manifest` artifacts.

Recorded result:

- serialized artifact + manifest: **110,093 → 54,646 bytes** (**−50.36%**);
- parameters: **23,138 → 11,042** (**−52.28%**);
- CPU batch-128 inference, five fresh-process probes: **47.05 → 23.11 ms mean** (compact/large **0.491×**);
- load/materialize: **7.85 → 5.86 ms mean**, but cache sensitivity was observed, so this is secondary;
- process RSS delta: **4.72 → 4.56 MB** (compact/large **0.966×**), so meaningful host-RAM reduction was **not demonstrated**;
- test PPL: **19.2784 large vs 18.9322 compact**.

The compact artifact executes the learned 2-block compiler directly and does not reconstruct the original 4-block model.

See `docs/RUNTIME_POC.md`, `scripts/reproduce/g7_confirmatory/runtime_poc.py`, and `results/reproduction/runtime_poc_seed4300_report.json`.

This is a bounded small-model CPU PoC, not evidence of universal GPU/LLM/runtime speedup.

## What remains worth doing

Only one conditional scientific closure task remains:

- **direct compositional-simplification replication on one clearly different architecture/task**, only if a stronger publication-level generalization/novelty claim is desired (GitHub Issue #2).

This replication is not required to preserve the current snapshot at its current claim scope.

Everything else should be treated as future work for interested researchers rather than required completion work.

## Current public documentation

- `docs/PUBLIC_SNAPSHOT.md`
- `docs/CORE_DISCOVERY.md`
- `docs/CLAIMS_AND_EVIDENCE.md`
- `docs/TRAINING_TIME_CONSOLIDATION.md`
- `docs/LATE_STAGE_FINDINGS.md`
- `docs/NEGATIVE_RESULTS.md`
- `docs/APPLICATIONS.md`
- `docs/RUNTIME_POC.md`
- `docs/OPEN_QUESTIONS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/PUBLICATION_NOTES.md`
- `docs/TERMINOLOGY.md`
- `docs/FAQ.md`

The repository should be read as an auditable research snapshot, not as a production-ready compression/runtime library.
