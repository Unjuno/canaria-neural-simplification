# Project status

**Current mode: pre-announcement hardening. The repository is not ready for broad public announcement.**

Issue #13 is the integrated readiness gate.

## What is already stable

The historical tag `v0.2.0-public-snapshot` remains frozen at commit `556dce21c7a5516a16780cb28d528d1ff3968e53`. It is a research-history boundary, not current announcement approval.

The reviewed baseline retains the bounded thesis of **task-conditioned compositional simplification under explicit operational rules**:

> Some learned spans can admit smaller task-preserving replacements when fitted as one composed input-output function than when simplified at implementation-component boundaries.

The strongest direct evidence remains the residual-MLP digits replication:

- fresh seeds `1200–1207`;
- exact learned replacement-parameter matching at each grid point;
- composed lower minimum passing budget in `8/8`;
- geometric composed/component-wise budget ratio `0.4823×`;
- validation selects the endpoint; test evaluation follows selection.

SmallViT replication remains retained with the disclosed caveat that test metrics are recorded for all candidates even though test is not a selection variable.

## Current blockers

### 1. Repository surface — Issue #16

The repository had too much version-number archaeology mixed into active navigation. A dedicated restructure now separates current evidence from the old v10–v25 sequence:

- active `results/` is limited to core, replication, training-time, Phase 2, and reproduction evidence;
- older versioned docs/results/scripts/environment records are preserved under `archives/research-history/`;
- historical review/release gate files are moved under `archives/reviews/` and `archives/releases/`;
- current indexes and audits are being updated around the new boundary.

This restructure is science-neutral: historical content is preserved rather than rewritten.

### 2. Full pinned-environment reproduction

The seed-1200 smoke test is insufficient. The complete residual-MLP fresh cohort (`1200–1207`) must be regenerated from a clean checkout under the pinned environment and checked against committed endpoints/statistics.

A first one-shot GitHub Actions attempt did not yield usable reproduction evidence, so this gate remains open. An execution/infrastructure failure must not be reported as a scientific reproduction failure.

### 3. External-validity evidence — Issue #15

Completed draft Phase 3 regression work passed its locked operational replacement-budget rule, but its teacher is weak in absolute predictive performance: confirmatory teacher test R² is approximately `0.112–0.255`.

Therefore it is not yet suitable as a strong announcement-level statement that the effect generalizes to regression.

Issue #15 defines a separate stronger-teacher control. The completed Phase 3 protocol/results must not be modified post hoc. Stage A changes only the teacher training recipe while holding dataset, split, model architecture, span, and replacement accounting fixed; test is excluded from teacher-recipe selection. If Stage A cannot produce a materially stronger validation regime, stop rather than changing architecture opportunistically.

### 4. Final integrated claim review

After Issues #15/#16 and the pinned reproduction gate are resolved, re-review as one surface:

- `README.md`;
- `docs/CLAIMS_AND_EVIDENCE.md`;
- `docs/PUBLICATION_NOTES.md`;
- `CITATION.cff`;
- release metadata;
- negative/correction boundaries.

## Phase 2 correction boundary

Supported within the declared testbed:

- Phase 2A: 4-bit composed coded-size result — `VALID_PASS`;
- Phase 2B: capacity-only rescue of naïve 3-bit per-matrix PTQ — `VALID_FAIL`;
- Phase 2C: row-wise scale rescue — `VALID_PASS` for both topologies;
- corrected later work supports viability of short activation-domain QAT-style repair in the tested residual-MLP family.

Critical invalidation:

**Phase 2E is `INVALIDATED_IMPLEMENTATION_BUG` and `DO_NOT_USE_FOR_INFERENCE`.**

Repair used raw digit inputs `Xt` instead of internal activation `ta[0]`; equal width 64 hid the semantic-domain error.

Consequences:

- Phase 2E `0/8` is not scientific negative evidence;
- Phase 2I's RNG causal explanation is retracted;
- 2H/2J interpretations tied to 2E are weakened/confounded;
- Phase 2O remains `VALID_UNCERTAIN`; no reliable composed repair-sample advantage is claimed.

## Training-time boundary

Retained within the tested small-model settings:

- G7 primary: progressive consolidation beat preregistered early/late one-shot controls;
- G15/G17: intervening task learning, not merely multiple compiler fits, is part of the tested staged advantage;
- G18: tested deadline-aware controller improved over the tested static controller;
- G20d/e and G22–G26: recontracting can make later fitting easier while residual error becomes more task-sensitive.

G21 remains a valid failure. G27 remains exploratory/no-Pareto-claim.

## Systems boundary

- G7 seed 4300 exact rerun is reproduction/portability evidence, not independent scientific replication.
- The CPU runtime/materialization result is one small PoC only.
- Meaningful host-RAM reduction, GPU/VRAM/energy/large-model/general runtime gains are not established.

## Research rule

Do not broaden a completed protocol after observing its outcomes. New work must have a new issue/phase, locked hypothesis/protocol or explicitly exploratory evidence class, inferential-unit policy, and stopping rule.

For current release decisions, follow `docs/ANNOUNCEMENT_READINESS.md`, Issue #13, Issue #15, and Issue #16.
