# Research roadmap — pre-announcement hardening

**Current state:** the repository has a historical v0.2.0 research snapshot and a reviewed evidence baseline, but it is **not yet ready for broad announcement**. The active gate is Issue #13 / `ANNOUNCEMENT_READINESS.md`.

## 1. Finish headline reproducibility

The strongest direct result on `main` is the residual-MLP digits cohort `1200–1207` with exact learned replacement-budget matching.

Before announcement:

1. install the pinned Python 3.11 CPU environment recorded under `scripts/reproduce/core_discovery_digits/`;
2. rerun all eight already-observed confirmatory seeds from a clean checkout;
3. compare per-seed selected budgets and reconstructed aggregate statistics with the committed summary;
4. retain a machine-readable reproduction report with commit/environment provenance.

This is reproduction evidence, not a new scientific cohort.

## 2. Resolve regression external-validity evidence

Draft Phase 3 changes task type from digits classification to diabetes regression and passed its locked operational budget rule. However, its confirmatory teacher test R² is weak (`~0.112–0.255`).

Do not change the completed Phase 3 protocol post hoc.

Before using regression as headline external-validity evidence, choose one of two paths:

- **bounded inclusion:** keep Phase 3 as evidence that the pattern appears in this weak-teacher tabular-regression regime and make that limitation prominent; or
- **new stronger-teacher test:** preregister a separate protocol using a better-performing regression teacher while keeping the replacement comparison as controlled as practical.

The second path is preferred if the goal is a stronger announcement-level external-validity statement.

## 3. Final integrated claim review

After the evidence set is fixed, reread together:

- `README.md`;
- `STATUS.md`;
- `CLAIMS_AND_EVIDENCE.md`;
- `PUBLICATION_NOTES.md`;
- `NEGATIVE_RESULTS.md`;
- `CITATION.cff`;
- release metadata.

The review must verify that no historical PASS, reproduction result, systems PoC, or invalidated experiment is promoted beyond its evidence class.

## 4. Final repository gate

Final candidate must pass:

```bash
python -m unittest discover -s tests -v
python tools/audit_repo.py
python tools/audit_readiness.py
```

and GitHub `repository-audit` under the pinned CPU environment.

Remove one-shot execution workflows from the final branch tip after their provenance/output is retained.

## Historical work that remains closed

The following should not be reopened merely for presentation:

- 2026-08-26 independent re-review and Phase 2E invalidation;
- v0.2.0 frozen tag boundary;
- G7 seed-4300 portability reproduction;
- bounded runtime PoC;
- completed G7–G27 protocols.

New evidence must use a new protocol rather than modifying a completed one after outcomes are known.

## Longer-term research after announcement readiness

Separate from the current gate:

- larger pretrained Transformer/LLM external validity;
- additional task types, spans, widths, and replacement grammars;
- codec-independent MDL/description measures;
- off-manifold functional complexity;
- effective repair/tangent dimension;
- mechanism dictionaries/algebra;
- sensitivity-aware utility-cost controllers;
- hardware-specific functional IR/JIT;
- larger-scale RAM/VRAM/energy/runtime benchmarking.
