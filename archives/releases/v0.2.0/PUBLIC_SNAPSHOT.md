# Canaria historical v0.2.0 research snapshot

This file explains how to read the frozen v0.2.0 research snapshot together with the later reviewed evidence state on `main`.

**Important:** the existence of this snapshot does **not** mean the current repository is ready for broad announcement. Current readiness is tracked in `ANNOUNCEMENT_READINESS.md` and Issue #13.

## Read order

1. `../README.md` — scoped project statement and current pre-announcement state.
2. `../STATUS.md` — current research/readiness status.
3. `ANNOUNCEMENT_READINESS.md` — active release/communication gate.
4. `../REPOSITORY_LAYOUT.md` — repository/evidence lifecycle and branch conventions.
5. `INDEPENDENT_REREVIEW_2026-08-26.md` — independent re-review decision ledger.
6. `CLAIMS_AND_EVIDENCE.md` — reviewed claim registry for the current baseline.
7. `CORE_DISCOVERY.md` — central operational compositional-simplification claim.
8. `CROSS_FAMILY_COMPOSITION_REPLICATION.md` — SmallViT direct replication and isolation caveat.
9. `CORE_DISCOVERY_REPLICATION_DIGITS.md` — residual-MLP exact-budget replication.
10. `TRAINING_TIME_CONSOLIDATION.md` and `LATE_STAGE_FINDINGS.md` — training-time evidence.
11. `phase2/README.md` — post-snapshot precision/quantization corrections.
12. `NEGATIVE_RESULTS.md` — valid negative evidence versus invalidated evidence.
13. `REPRODUCIBILITY.md` — integrity/reproduction policy.
14. `PUBLICATION_NOTES.md` — communication-safe wording, not approval to announce.
15. `RELEASE_CHECKLIST.md` — historical v0.2.0 snapshot/version-control record.
16. `HISTORICAL_INDEX.md` — preserved historical material.

## Frozen version boundary

The immutable historical reference is tag `v0.2.0-public-snapshot`, pointing to commit:

`556dce21c7a5516a16780cb28d528d1ff3968e53`

The associated GitHub release was created as a research snapshot. Later evidence and corrections are not back-projected into that tag.

The 2026-08-26 re-review and PR #7 correction merge are later repository history. They improve the evidence surface but do not, by themselves, certify announcement readiness.

## Snapshot interpretation

Canaria is an auditable research record, not a production-ready compression library.

The retained core statement is operational:

> under explicit task distributions, replacement grammars, and passing criteria, some learned spans in the tested networks admit smaller task-preserving replacements when fitted as composed input-output functions than when simplified at implementation-component boundaries.

The repository does **not** claim mathematical/Kolmogorov complexity reduction or universal Transformer/LLM behavior.

## Direct architecture-family evidence

### Residual MLP — strongest matched-budget baseline experiment

Fresh seeds `1200–1207` used exact learned replacement-parameter matching at every budget point.

- component-wise mean minimum passing budget: `3584`;
- composed mean minimum passing budget: `1728`;
- composed smaller: `8/8`;
- geometric budget ratio: `0.4823×`;
- selected-budget test-accuracy difference: `+0.583` percentage points, bootstrap95 `[+0.306,+0.806]` pt.

Validation selects the endpoint; test evaluation follows selection.

The 2048-parameter joint-factorized control is descriptive/mechanistic secondary, not a confirmatory causal decomposition.

Announcement hardening adds a pinned-environment full-cohort reproduction requirement; see `ANNOUNCEMENT_READINESS.md`.

### Small Vision Transformer

Under the locked rule across 8 fresh eligible seeds:

- component-wise selected replacement: `9808` parameters;
- composed selected replacement: `4904–5424`;
- mean ratio: `0.51988`;
- bootstrap95 `[0.50634,0.53926]`;
- composed smaller: `8/8`.

Independent re-review found an important isolation boundary: the selection criterion excludes test accuracy, but the runner records test metrics for every candidate. Therefore say **“test was not a selection variable”**, not “test remained operationally hidden until after selection.”

## Training-time evidence boundary

The small real-text character-LM program is retained with evidence tiers explicit:

- G7 primary PASS is progressive versus early/late one-shot;
- small-from-start and large-reference differences are secondary observations;
- G15/G17 support an intervening-learning/recontracting interpretation under the tested schedules;
- G18 supports the specific tested deadline-aware controller comparison, not a universal controller rule;
- G21 remains a valid failure;
- G27 remains exploratory without a Pareto claim.

Compiler cost in these experiments is generally a proxy, not measured hardware energy/FLOPs/wall-clock equivalence.

## Phase 2 correction

Phase 2A–C remain usable under their declared residual-MLP quantizer/accounting.

Phase 2E is **`INVALIDATED_IMPLEMENTATION_BUG`** and `DO_NOT_USE_FOR_INFERENCE` because repair used raw `Xt` where the replacement was defined on internal activation `ta[0]`; equal width 64 hid the semantic error.

The invalid result is retained as history, not negative scientific evidence. Phase 2I's RNG causal explanation is retracted. Phase 2O did not confirm a reliable composed repair-sample advantage.

Not all later 2D–2O raw per-seed artifacts are checked into Git. Their correction archive is identified by SHA256:

`1a339be12d7644de534ac77a712307c49ee0c3d9acb28c8a3532883edca3dab7`

See `phase2/README.md` and `../results/phase2/precision_composition/INVALIDATED_HISTORY.md`.

## Reproduction boundary

A self-contained G7 seed-4300 runner exactly matched the archived confirmatory JSON in its recorded environment:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

This is reproduction/portability evidence for an already-confirmatory seed, not a new independent scientific replication.

The residual-MLP digits headline experiment has a separate pinned-environment cohort-reproduction gate during pre-announcement hardening.

## Runtime PoC boundary

One small CPU-only PoC reported:

- serialized artifact + manifest: `110,093 → 54,646 bytes`;
- batch-128 CPU inference mean: `47.05 → 23.11 ms`;
- meaningful host-RAM reduction: **not demonstrated**.

Do not generalize this to GPU/VRAM/energy/large models or universal runtime improvement.

## Historical closure versus current readiness

The v0.2.0 version-control sequence was completed: independent review, frozen tag/release boundary, correction merge, stale-branch cleanup, and repository audit.

That sequence is a **historical snapshot record**, not the current announcement gate. Current work must satisfy `ANNOUNCEMENT_READINESS.md`. Candidate later experiments remain isolated until an explicit scientific inclusion/exclusion decision is made.