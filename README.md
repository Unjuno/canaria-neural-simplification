# Canaria Neural Simplification

**Canaria** studies a simple empirical pattern:

> A learned span can sometimes be represented more simply when several learned computations are treated as one composed input-output function rather than simplified separately at implementation-block boundaries.

```text
x ── f ──> h ── g ──> y

separate: simplify(f) + simplify(g)
composed: simplify(g ∘ f)
```

This repository is currently at a **pre-publication independent re-review gate**. The frozen v0.2.0 **public snapshot** remains the baseline; post-snapshot precision work is staged separately.

For the final review procedure, see [`REVIEW_HANDOFF.md`](REVIEW_HANDOFF.md). Unsupported public claims should be removed or narrowed before publication; invalidated raw evidence is preserved with explicit correction markers.

## Try the core pattern

A one-seed residual-MLP direct experiment is available:

```bash
python -m pip install numpy torch scikit-learn
python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 \
  --out /tmp/canaria_seed1200.json
```

Recorded seed 1200 selected:

- component-wise replacement: `3072` parameters
- composed replacement: `1536` parameters

See [`QUICKSTART.md`](QUICKSTART.md) for the minimal reproduction path.

## Evidence at a glance

Current retained evidence, pending final independent re-review:

- direct component-wise vs composed replication in a Small Vision Transformer;
- direct component-wise vs composed replication in a residual MLP with matched learned replacement-parameter budgets;
- training-time consolidation/recontracting experiments with positive and negative controls;
- bounded CPU serialization/runtime proof of concept;
- precision/quantization follow-up showing that bit width alone is not enough to describe deployable complexity.

The repository does **not** claim universal mathematical/Kolmogorov complexity reduction, LLM-scale validity, or general RAM/GPU/energy/runtime improvement.

## Important Phase 2 correction

Phase 2E is **INVALIDATED_IMPLEMENTATION_BUG**. Its repair path used raw `Xt` instead of the intended internal activation domain `ta[0]`; the equal width made the bug silent.

The invalid evidence is retained for provenance but is not used as support. See:

- [`docs/phase2/README.md`](docs/phase2/README.md)
- [`results/phase2/precision_composition/CORRECTION_STATUS.json`](results/phase2/precision_composition/CORRECTION_STATUS.json)

## Where to look next

- [`QUICKSTART.md`](QUICKSTART.md) — run one minimal direct experiment.
- [`REVIEW_HANDOFF.md`](REVIEW_HANDOFF.md) — independent pre-publication review procedure.
- [`docs/CORE_DISCOVERY.md`](docs/CORE_DISCOVERY.md) — central empirical claim and scope.
- [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md) — claim registry.
- [`docs/phase2/README.md`](docs/phase2/README.md) — precision/quantization evidence and correction status.
- [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) — preserved failed hypotheses.
- [`STATUS.md`](STATUS.md) — current stopping and release policy.

## Repository layout

- `scripts/reproduce/` — portable minimal/public reproduction runners.
- `scripts/replication/` — fresh direct replication runners.
- `scripts/phase2/` — post-snapshot precision experiments.
- `results/` — protocol locks and machine-readable evidence.
- `docs/` — interpretation, boundaries, corrections, and history.
- `archives/` — retained handoff/history material.

## License

Original code and documentation are released under the **Apache License 2.0**. Third-party datasets and libraries remain under their own licenses.

## Citation

See [`CITATION.cff`](CITATION.cff). Until a paper is published, cite the exact repository commit/snapshot plus the protocol/result files used.