# Canaria Neural Simplification

**Canaria** studies a bounded empirical pattern:

> Under explicit task distributions, replacement grammars, and passing criteria, some learned spans admit smaller task-preserving replacements when fitted as one composed input-output function than when simplified at implementation-component boundaries.

```text
x ── f ──> h ── g ──> y

component-wise: simplify(f) + simplify(g)
composed:       simplify(g ∘ f)
```

This is an **operational replacement/description-complexity** claim. The repository does not claim universal mathematical/Kolmogorov complexity reduction.

## Research status — pre-announcement

**This repository is not currently announcement-ready.**

A historical v0.2.0 research snapshot exists at tag [`v0.2.0-public-snapshot`](https://github.com/Unjuno/canaria-neural-simplification/releases/tag/v0.2.0-public-snapshot), pointing to commit `556dce21c7a5516a16780cb28d528d1ff3968e53`. That snapshot records a past review/version boundary; it should not be read as a statement that the present project is ready for broad public announcement.

Current hardening is tracked in Issue #13. The remaining gate includes pinned clean-clone reproduction of the headline direct experiment, a final decision on candidate external-validity evidence, and another claim/readiness review after those choices are made.

The 2026-08-26 independent re-review remains relevant:

- [`docs/INDEPENDENT_REREVIEW_2026-08-26.md`](docs/INDEPENDENT_REREVIEW_2026-08-26.md)
- [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md)

`main` is the reviewed evidence baseline used for hardening. New research on separate branches or draft PRs is not part of the headline claim set until separately reviewed and merged.

See [`docs/ANNOUNCEMENT_READINESS.md`](docs/ANNOUNCEMENT_READINESS.md) for the current gate.

## Try the representative direct experiment

For a convenience run:

```bash
python -m pip install numpy torch scikit-learn
python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 \
  --out /tmp/canaria_seed1200.json
```

For **reproduction evidence**, do not rely on latest-package resolution; use the pinned environment described in [`scripts/reproduce/core_discovery_digits/README.md`](scripts/reproduce/core_discovery_digits/README.md).

Recorded seed 1200:

- component-wise minimum passing budget: `3072` learned replacement parameters;
- composed minimum passing budget: `1536`.

Across fresh seeds `1200–1207`:

- component-wise mean minimum passing budget: `3584`;
- composed mean minimum passing budget: `1728`;
- composed lower: `8/8`;
- geometric composed/component-wise budget ratio: `0.4823×`.

The residual-MLP runner exactly matches learned replacement-parameter budgets at each grid point, uses validation to select the minimum passing budget, and evaluates test utility only after endpoint selection.

See [`QUICKSTART.md`](QUICKSTART.md).

## Evidence at a glance

Retained after independent re-review:

- residual-MLP direct component-wise/composed replication with exact learned-budget matching;
- SmallViT direct component-wise/composed replication under a locked rule, with an explicit caveat that its runner records test metrics for all candidates even though test is not a selection variable;
- bounded training-time consolidation/recontracting experiments with positive and negative controls;
- one bounded CPU serialization/direct-execution proof of concept;
- Phase 2A–C precision/quantization evidence plus explicit correction history for later Phase 2 work.

The repository does **not** claim universal LLM-scale validity or general FLOP, RAM, VRAM, GPU, energy, wall-clock, or runtime improvement.

Candidate post-snapshot research may broaden this evidence, but it is not promoted into the headline claim set merely because an individual locked protocol passed.

## Critical Phase 2 correction

Phase 2E is **`INVALIDATED_IMPLEMENTATION_BUG`** and `DO_NOT_USE_FOR_INFERENCE`.

Its repair path used raw `Xt` instead of the intended internal activation domain `ta[0]`; equal width 64 made the semantic error silent.

The invalid result is preserved as correction history, not scientific evidence:

- [`docs/phase2/README.md`](docs/phase2/README.md)
- [`results/phase2/precision_composition/CORRECTION_STATUS.json`](results/phase2/precision_composition/CORRECTION_STATUS.json)
- [`results/phase2/precision_composition/INVALIDATED_HISTORY.md`](results/phase2/precision_composition/INVALIDATED_HISTORY.md)

Phase 2O did **not** confirm a reliable compositional repair-sample advantage; that positive claim is removed.

## Reproducibility and systems boundary

A portable G7 seed-4300 runner exactly reproduced the archived JSON in its recorded environment. This is software/portability evidence for an already-confirmatory seed, **not** a new independent scientific replication.

The headline residual-MLP direct experiment now has a dedicated pinned-environment reproduction path. Announcement readiness requires the full fresh cohort (`1200–1207`), not only the seed-1200 smoke test, to be regenerated and checked under that path.

One small CPU PoC produced a smaller serialized artifact and lower measured batch-128 CPU inference latency. Meaningful host-RAM reduction was not demonstrated; GPU/VRAM/energy/large-model generalization remains open.

## Repository map

The repository deliberately separates reviewed interpretation, immutable evidence, historical experiment code, and reusable code. See [`REPOSITORY_LAYOUT.md`](REPOSITORY_LAYOUT.md) before reorganizing paths or interpreting a versioned directory as current by name alone.

- `docs/` — current interpretation, review records, protocols, and preserved history.
- `results/` — machine-readable evidence and correction records.
- `scripts/` — reproduction, replication, phase-specific, and historical experiment runners.
- `src/canaria/` — reusable cleaned code; currently distinct from historical experiment scripts.
- `tools/` — repository/evidence audits.

## Where to look

- [`docs/ANNOUNCEMENT_READINESS.md`](docs/ANNOUNCEMENT_READINESS.md) — current release/announcement blockers and exit criteria.
- [`QUICKSTART.md`](QUICKSTART.md) — minimal direct experiment.
- [`REPOSITORY_LAYOUT.md`](REPOSITORY_LAYOUT.md) — directory, evidence-lifecycle, branch, and workflow conventions.
- [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md) — reviewed claim registry for the current baseline.
- [`docs/INDEPENDENT_REREVIEW_2026-08-26.md`](docs/INDEPENDENT_REREVIEW_2026-08-26.md) — independent decision ledger.
- [`docs/CORE_DISCOVERY.md`](docs/CORE_DISCOVERY.md) — central empirical claim and scope.
- [`docs/phase2/README.md`](docs/phase2/README.md) — precision/quantization corrections.
- [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) — valid negative evidence versus invalidated evidence.
- [`STATUS.md`](STATUS.md) — current hardening state.

## License

Original code and documentation are released under the **Apache License 2.0**. Third-party datasets and libraries remain under their own licenses.

## Citation

See [`CITATION.cff`](CITATION.cff). Until a paper is published and the announcement-readiness gate is closed, cite the exact repository commit/snapshot and the protocol/result artifacts supporting the specific claim.