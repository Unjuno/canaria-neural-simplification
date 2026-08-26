# Canaria Neural Simplification

**Canaria** studies a bounded empirical pattern:

> Under explicit task distributions, replacement grammars, and passing criteria, some learned spans admit smaller task-preserving replacements when fitted as one composed input-output function than when simplified at implementation-component boundaries.

```text
x ── f ──> h ── g ──> y

component-wise: simplify(f) + simplify(g)
composed:       simplify(g ∘ f)
```

This is an **operational replacement/description-complexity** claim. The repository does not claim universal mathematical/Kolmogorov complexity reduction.

The frozen v0.2.0 **public snapshot** remains the version baseline. Post-snapshot precision work is staged on this research branch.

## Pre-publication review state

The independent re-review required by `REVIEW_HANDOFF.md` and Issue #9 was performed on 2026-08-26. The decision ledger is:

- [`docs/INDEPENDENT_REREVIEW_2026-08-26.md`](docs/INDEPENDENT_REREVIEW_2026-08-26.md)
- [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md)

Do not publish/post or mark PR #7 ready while Issue #9 remains open. Closing that issue completes the scientific quality gate only; the v0.2.0 release/tag boundary and PR #7 merge remain separate decisions.

## Try the strongest minimal public experiment

```bash
python -m pip install numpy torch scikit-learn
python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 \
  --out /tmp/canaria_seed1200.json
```

Recorded seed 1200:

- component-wise minimum passing budget: `3072` learned replacement parameters;
- composed minimum passing budget: `1536`.

Across fresh seeds `1200–1207`:

- component-wise mean minimum passing budget: `3584`;
- composed mean minimum passing budget: `1728`;
- composed lower: `8/8`;
- geometric composed/component-wise budget ratio: `0.4823×`.

This residual-MLP runner exactly matches learned replacement-parameter budgets at each grid point, uses validation to select the minimum passing budget, and evaluates test utility only after endpoint selection.

See [`QUICKSTART.md`](QUICKSTART.md).

## Evidence at a glance

Retained after independent re-review:

- residual-MLP direct component-wise/composed replication with exact learned-budget matching;
- SmallViT direct component-wise/composed replication under a locked rule, with an explicit caveat that its runner records test metrics for all candidates even though test is not a selection variable;
- bounded training-time consolidation/recontracting experiments with positive and negative controls;
- one bounded CPU serialization/direct-execution proof of concept;
- Phase 2A–C precision/quantization evidence plus explicit correction history for later Phase 2 work.

The repository does **not** claim universal LLM-scale validity or general FLOP, RAM, VRAM, GPU, energy, wall-clock, or runtime improvement.

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

One small CPU PoC produced a smaller serialized artifact and lower measured batch-128 CPU inference latency. Meaningful host-RAM reduction was not demonstrated; GPU/VRAM/energy/large-model generalization remains open.

## Where to look

- [`QUICKSTART.md`](QUICKSTART.md) — minimal direct experiment.
- [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md) — authoritative public claim registry.
- [`docs/INDEPENDENT_REREVIEW_2026-08-26.md`](docs/INDEPENDENT_REREVIEW_2026-08-26.md) — independent decision ledger.
- [`docs/CORE_DISCOVERY.md`](docs/CORE_DISCOVERY.md) — central empirical claim and scope.
- [`docs/phase2/README.md`](docs/phase2/README.md) — precision/quantization corrections.
- [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) — valid negative evidence versus invalidated evidence.
- [`STATUS.md`](STATUS.md) — publication/release state.

## License

Original code and documentation are released under the **Apache License 2.0**. Third-party datasets and libraries remain under their own licenses.

## Citation

See [`CITATION.cff`](CITATION.cff). Until a paper is published, cite the exact repository commit/snapshot and the protocol/result artifacts supporting the specific claim.
