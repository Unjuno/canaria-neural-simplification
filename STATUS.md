# Project status

**Current mode: pre-announcement hardening. The repository is not yet ready for broad public announcement.**

## Historical snapshot versus current readiness

- Historical tag: `v0.2.0-public-snapshot`.
- Historical snapshot commit: `556dce21c7a5516a16780cb28d528d1ff3968e53`.
- The 2026-08-26 independent re-review and subsequent correction merge remain part of the evidence history.
- `main` is the reviewed evidence baseline from which current hardening proceeds.
- The existence of the historical GitHub release/tag does **not** mean the current project has passed an announcement-readiness gate.

Issue #13 is the active readiness gate. Do not describe the repository as announcement-ready until that issue is closed after a final claim/reproducibility review.

## Core claim retained on the reviewed baseline

The project-level thesis remains **task-conditioned compositional simplification under explicit operational rules**:

> Some learned spans can admit smaller task-preserving replacements when fitted as one composed input-output function than when simplified at implementation-component boundaries.

The strongest direct result currently on `main` is the residual MLP with exact learned replacement-parameter matching:

- fresh seeds `1200–1207`;
- composed lower minimum passing budget in `8/8`;
- geometric composed/component-wise budget ratio `0.4823×`;
- validation selects the endpoint; test evaluation follows selection.

The SmallViT direct replication is retained with a disclosed isolation caveat: its locked selection rule excludes test metrics, but its runner records test accuracy for all candidates. That is weaker operational test isolation than the residual-MLP runner.

## Current announcement blockers

### 1. Full pinned-environment reproduction of the headline cohort

The seed-1200 smoke test is insufficient as the final public reproducibility gate. The complete fresh residual-MLP cohort (`1200–1207`) must be regenerated from a clean checkout using the pinned reproduction environment and checked against the committed primary endpoints/statistics.

### 2. External-validity evidence selection

Draft Phase 3 regression work is not automatically part of the headline claim set. Its protocol/result audit is valid as a completed experiment, but its teacher is weak in absolute predictive performance (confirmatory test R² approximately `0.112–0.255`). Before broad communication, either:

- retain it as a bounded weak-teacher external-validity result with that limitation prominent; or
- run a separately locked stronger-teacher regression experiment and decide which evidence belongs in the final claim set.

Do not modify the completed Phase 3 protocol post hoc.

### 3. Final communication review

After evidence inclusion/exclusion is settled, re-review `README.md`, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/PUBLICATION_NOTES.md`, `CITATION.cff`, and the announcement-readiness checklist as one surface.

## Training-time boundary

Retained:

- G7 primary: progressive consolidation beat preregistered early/late one-shot controls;
- G15/G17: intervening task learning, not merely factorized fitting, is part of the tested staged advantage;
- G19: staged-path effect also observed on `5→4→2` versus `5→2` with equal compiler-update counts;
- G18: the tested deadline-aware controller improved over the tested static controller;
- G20d/e and G22–G26: recontracting can make fitting easier while residual error becomes more task-sensitive in the small character-LM testbed.

G21 remains a valid failure and G27 remains exploratory/no-Pareto-claim.

## Phase 2 correction boundary

Supported:

- Phase 2A: 4-bit composed coded-size advantage under the locked residual-MLP quantizer/accounting;
- Phase 2B: increasing weight count alone did not rescue the tested naive 3-bit per-matrix PTQ;
- Phase 2C: row-wise scales rescued 3-bit PTQ for both topologies;
- corrected later work supports viability of short activation-domain QAT-style repair in the tested residual-MLP family.

Critical invalidation:

**Phase 2E is `INVALIDATED_IMPLEMENTATION_BUG` and `DO_NOT_USE_FOR_INFERENCE`.**

Its repair code used raw digit inputs `Xt` where the replacement was defined on internal activation `ta[0]`. Equal width 64 made the semantic error silent.

Consequences:

- Phase 2E `0/8` is not scientific negative evidence;
- Phase 2I's RNG causal explanation is retracted;
- 2H/2J interpretations tied to 2E are weakened/confounded;
- Phase 2O did not confirm a reliable composed repair-sample advantage (`UNCERTAIN`).

Invalidation history is preserved in:

- `results/phase2/precision_composition/CORRECTION_STATUS.json`
- `results/phase2/precision_composition/INVALIDATED_HISTORY.md`

Not all later 2D–2O raw per-seed artifacts are checked into Git; the correction archive is identified by SHA256 `1a339be12d7644de534ac77a712307c49ee0c3d9acb28c8a3532883edca3dab7`.

## Reproducibility / systems boundary

- G7 seed 4300 exact portable rerun: reproduction/portability evidence, not independent replication.
- Residual-MLP digits: headline scientific result with a new pinned-environment full-cohort reproduction gate in progress.
- Runtime PoC: one small CPU/storage/inference result only.
- Meaningful host-RAM reduction, GPU/VRAM/energy/large-model/general runtime gains: not established.

## Repository / research-state boundary

Repository organization follows `REPOSITORY_LAYOUT.md`:

- `main` contains the reviewed baseline plus readiness hardening once merged;
- unmerged research branches and draft PRs are work-in-progress, not automatic claim updates;
- maintenance changes should remain science-neutral and separate from fresh experimental outcomes;
- historical protocols/results are preserved rather than rewritten to match later interpretation.

`repository-audit` is an integrity check, not by itself an announcement-readiness certificate.

## Stopping rule

Do not broaden a completed protocol after seeing its outcomes. New scientific work must use a new issue/research phase with its own hypothesis, protocol/evidence class, inferential-unit policy, and stopping rule.

For the current release decision, follow `docs/ANNOUNCEMENT_READINESS.md` and Issue #13.