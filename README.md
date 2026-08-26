# Canaria Neural Simplification

Can a learned computation become **simpler when we stop respecting the implementation boundaries that produced it**?

That is the core question in this repository.

```text
x ── f ──> h ── g ──> y

separate treatment:   simplify(f) + simplify(g)
composed treatment:   simplify(g ∘ f)
```

In several small-model experiments, the second route needed substantially less task-preserving replacement capacity than simplifying the two implementation blocks separately.

This repository is an auditable research handoff: a minimal runnable experiment first, machine-readable evidence next, and the full exploration history behind that.

## Run the smallest direct experiment

```bash
python -m pip install numpy torch scikit-learn
python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 \
  --out /tmp/canaria_seed1200.json
```

Recorded confirmatory seed `1200` selected:

```text
component-wise minimum passing budget: 3072 parameters
composed minimum passing budget:       1536 parameters
ratio:                                 0.5×
```

The locked 8-seed residual-MLP replication found a geometric mean composed/component-wise budget ratio of **0.4823×**, with the composed condition smaller in **8/8 seeds**. A separate Small Vision Transformer replication also found the composed replacement smaller in **8/8 fresh seeds**.

See [`QUICKSTART.md`](QUICKSTART.md) for the shortest path from clone to result.

## What to look at

The main empirical pattern is not “compression always works.” It is narrower:

> An implementation decomposition can overstate the task-effective complexity of a learned span. Directly approximating the composed input-output map can sometimes expose a smaller representation.

Useful entry points:

- [`docs/CORE_DISCOVERY.md`](docs/CORE_DISCOVERY.md) — original observation and claim boundary.
- [`docs/CORE_DISCOVERY_REPLICATION_DIGITS.md`](docs/CORE_DISCOVERY_REPLICATION_DIGITS.md) — residual-MLP direct replication and functional-boundary control.
- [`docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md`](docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md) — SmallViT direct replication.
- [`docs/phase2/README.md`](docs/phase2/README.md) — precision/quantization follow-up and correction record.
- [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md) — claim registry.
- [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) — failed hypotheses kept as evidence.
- [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md) — places where another researcher can extend the work.

## One mechanistic clue

At a fixed 2048-parameter budget in the residual-MLP replication:

```text
fit each block locally                         NMSE 0.1474
same two-module topology, fit end-to-end span NMSE 0.0639
one composed module, fit end-to-end span      NMSE 0.0533
```

Most of the gain appeared when the **training objective/boundary** changed from local blocks to the composed span, even before changing the two-module topology. That is one reason the repository treats functional boundaries as a first-class variable.

## Precision follow-up

The effect survived a locked 4-bit post-training-quantization experiment on the residual-MLP setup. At 3 bit, naive per-matrix PTQ failed, but row-wise scales or a short correctly implemented quantization-aware repair could recover fidelity in this model family.

A later audit found a silent input-domain bug in one repair experiment (Phase 2E). The invalid result is preserved and explicitly marked rather than deleted. The correction and replacement experiments are documented in [`docs/phase2/README.md`](docs/phase2/README.md).

## Scope

This repository does **not** establish universal mathematical/Kolmogorov complexity reduction, universal neural-network simplification, large-LLM validity, or guaranteed GPU/RAM/energy/runtime gains.

What it does provide is a set of falsifiable small-model observations, direct replications, negative results, portable runners, protocol locks, and machine-readable evidence around task-conditioned compositional simplification.

## Repository map

```text
scripts/reproduce/   shortest runnable reproductions
scripts/replication/ fresh direct replication runners
scripts/phase2/      post-snapshot precision experiments
results/             protocol locks and machine-readable evidence
docs/                interpretation, boundaries, and research record
src/canaria/         reusable cleaned components
archives/            retained historical/handoff material
```

For the current stopping/release state, see [`STATUS.md`](STATUS.md).

## License and citation

Original code and documentation are released under the **Apache License 2.0**. Third-party datasets and libraries remain under their own licenses.

See [`CITATION.cff`](CITATION.cff). Until a paper is published, cite the exact repository commit plus the protocol/result files used.