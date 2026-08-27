# Canaria Neural Simplification

Canaria studies a bounded empirical question:

> Under an explicit task distribution, replacement grammar, parameter accounting rule, and validation criterion, can a learned multi-component span be replaced more compactly when fitted as one composed input-output function than when its implementation components are replaced separately?

This is an **operational replacement/description-complexity** question. It is not a claim about universal mathematical or Kolmogorov complexity.

## Status: pre-announcement

**This repository is not currently ready for broad announcement.**

A historical tag, `v0.2.0-public-snapshot`, exists at commit `556dce21c7a5516a16780cb28d528d1ff3968e53`. It is an immutable research-history boundary, not a certificate that the current project is announcement-ready.

Current hardening is tracked in Issue #13. The active gate is [`docs/ANNOUNCEMENT_READINESS.md`](docs/ANNOUNCEMENT_READINESS.md).

## Evidence map

| Evidence line | Current interpretation |
| --- | --- |
| residual MLP, sklearn digits, seeds 1200–1207 | reviewed direct matched-budget replication; composed minimum validation-passing replacement budget was lower in 8/8 seeds |
| SmallViT, sklearn digits | reviewed direct cross-architecture replication; retained with an explicit test-recording caveat |
| training-time consolidation/recontracting | bounded positive and negative evidence within the tested training regimes |
| Phase 2 precision/quantization | reviewed Phase 2A–C evidence plus explicit correction history; later claims are more limited |
| recursive Canaria composition | integrated C-series research line: joint boundary re-alignment, recursive recompilation, depth-2/depth-3 confirmation, and self-anchored compressed-interface controls; not automatically in the headline set |
| systems/runtime | integrated S1–S7 evidence: learned streaming, memory-ceiling control, independent/native execution, and a 43,808-B explicit application-arena result for one locked prototype; not a physical-MCU claim |

The reviewed baseline claim registry is [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md). Current research lines have their own strict maps:

- [`docs/RECURSIVE_COMPOSITION.md`](docs/RECURSIVE_COMPOSITION.md)
- [`docs/SYSTEMS_RUNTIME.md`](docs/SYSTEMS_RUNTIME.md)

Passing research protocols do not enter the final announcement claim set until the release/readiness review explicitly promotes them.

## Representative reproduction

A convenience single-seed run is:

```bash
python -m pip install numpy torch scikit-learn
python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 \
  --out /tmp/canaria_seed1200.json
```

For actual reproduction evidence, use the pinned environment in [`scripts/reproduce/core_discovery_digits/README.md`](scripts/reproduce/core_discovery_digits/README.md), not unconstrained latest packages.

Recorded seed 1200:

- component-wise minimum passing budget: `3072` learned replacement parameters;
- composed minimum passing budget: `1536`.

Recorded fresh cohort 1200–1207:

- component-wise mean minimum passing budget: `3584`;
- composed mean minimum passing budget: `1728`;
- composed lower: `8/8`;
- geometric composed/component-wise budget ratio: `0.4823×`.

Announcement readiness requires the full cohort to be regenerated and checked under the pinned clean-clone path; a one-seed smoke test is not sufficient.

See [`QUICKSTART.md`](QUICKSTART.md).

## Critical correction

Phase 2E is **`INVALIDATED_IMPLEMENTATION_BUG`** and **`DO_NOT_USE_FOR_INFERENCE`**. The repair path used raw `Xt` instead of the intended internal activation `ta[0]`; the equal 64-dimensional width hid the semantic-domain error.

The invalid result is retained as provenance, not evidence:

- [`docs/phase2/README.md`](docs/phase2/README.md)
- [`results/phase2/precision_composition/CORRECTION_STATUS.json`](results/phase2/precision_composition/CORRECTION_STATUS.json)
- [`results/phase2/precision_composition/INVALIDATED_HISTORY.md`](results/phase2/precision_composition/INVALIDATED_HISTORY.md)

Phase 2O remains `VALID_UNCERTAIN`; no reliable composed-repair sample-complexity advantage is claimed.

## Repository structure

The active surface is deliberately small:

- `docs/` — current interpretation, evidence summaries, readiness, corrections, and reference material;
- `results/` — current machine-readable evidence and reproduction/correction records;
- `results/recursive_composition/` — integrated C-series composition evidence;
- `results/systems/` — integrated S1–S7 runtime evidence;
- `scripts/reproduce/` — supported clean-clone reproduction paths;
- `scripts/replication/` — direct replication runners;
- `scripts/recursive_composition/` and `scripts/systems/` — current research-line runners retained for audit/reuse;
- `scripts/phase2/` and `scripts/phases/training_time/` — active evidence-producing research code still referenced by reviewed results;
- `src/canaria/` — reusable cleaned code;
- `tools/` — integrity/readiness audits;
- `archives/` — historical release/review records and the older versioned research sequence.

The old v10–v25 experiment sequence is preserved under `archives/research-history/`; it is not first-class current navigation. See [`REPOSITORY_LAYOUT.md`](REPOSITORY_LAYOUT.md) and [`archives/README.md`](archives/README.md).

## What this repository does not establish

Current evidence does not establish:

- universal subadditivity of mathematical/Kolmogorov complexity;
- general LLM-scale validity;
- arbitrary or lossless recursive composition;
- arbitrary-subspace invariance or a universal minimum hidden-interface dimension;
- general RAM/VRAM/GPU/energy improvement or a demonstrated physical 44-KiB device;
- a universal quantization or repair advantage;
- that every passing exploratory or draft experiment belongs in the final announcement claim set.

## Reading order

1. [`docs/ANNOUNCEMENT_READINESS.md`](docs/ANNOUNCEMENT_READINESS.md)
2. [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md)
3. [`docs/CORE_DISCOVERY.md`](docs/CORE_DISCOVERY.md)
4. [`docs/RECURSIVE_COMPOSITION.md`](docs/RECURSIVE_COMPOSITION.md)
5. [`docs/SYSTEMS_RUNTIME.md`](docs/SYSTEMS_RUNTIME.md)
6. [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md)
7. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)

## License and citation

Original code and documentation are released under Apache License 2.0. Third-party datasets and libraries retain their own licenses.

Until a paper is published and the announcement-readiness gate closes, cite the exact repository commit/snapshot together with the protocol/result artifacts supporting the specific claim. See [`CITATION.cff`](CITATION.cff).