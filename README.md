# Canaria Neural Simplification

**Canaria** is an experimental research repository about a simple question:

> Can trained neural computation sometimes be represented more simply when several learned computations are treated as one composed function rather than as separate implementation blocks?

The repository preserves positive, negative, exploratory, confirmatory, and reproduction evidence. It is currently in a **research consolidation / public-snapshot phase**: broad experiment expansion is paused, and new experiments should be added only when they close a concrete evidential gap.

For the intended public reading order and snapshot policy, see [`docs/PUBLIC_SNAPSHOT.md`](docs/PUBLIC_SNAPSHOT.md).

## Core empirical finding

The strongest project-level result is **compositional simplification of learned computation**.

In the tested settings:

- simplification was not confined to high-Canary regions;
- composition complexity was frequently subadditive (`P(G>0)=0.7107`, 95% CI `0.6128–0.8137`) under the original confirmatory grammar;
- implementation-block boundaries were not always the most natural functional boundaries;
- a learned span could sometimes admit a smaller task-conditioned replacement even when component-wise simplification was poor;
- whole-network accounting confirmed that local simplification was not merely hidden parameter relocation under the measured codecs.

The careful claim is **not** that mathematical function complexity always decreases under composition. The supported claim is operational: for some trained networks and task distributions, the composed input-output map admits a substantially smaller task-preserving representation than component-wise treatment suggests.

See [`docs/CORE_DISCOVERY.md`](docs/CORE_DISCOVERY.md).

## Dynamic extension: consolidation during learning

Later experiments showed that simplification is not only post-hoc.

A useful working process is:

> **form → transfer → commit → recontract → transfer again**

A larger model forms useful computation; a span is transferred into a smaller replacement; task learning resumes; later consolidations may become easier.

Headline evidence:

- **G7:** progressive `4→3→2` consolidation reached a final model with **52.28% fewer parameters** and beat early/late one-shot controls.
- **G15 + G17:** staged consolidation helped only when task learning occurred between commits; merely factorizing one compiler fit into two did not reproduce the benefit.
- **G18:** a deadline-aware controller improved mean PPL by **0.181** versus the static controller while reducing mean compiler updates from **184 to 136** on fresh `n=12`.
- **G19:** the staged effect replicated on a different path, `5→4→2` versus direct `5→2`, by **−0.689 PPL** with identical compiler-update budgets, 8/8 fresh seeds.
- **G20d/G20e:** after recontracting, the next compiler reached the same normalized functional-error target with about **22% fewer updates**, but the same normalized error caused **more immediate task damage**.
- **G22–G26:** downstream sensitivity, error direction, logit-space second-order error, and remaining learning horizon materially improved task-damage prediction.
- **G21:** a hard shadow-damage veto failed because it prevented final contraction in 2/12 fresh seeds.
- **G27:** fixed risk caps exposed a cost/utility trade-off rather than a Pareto improvement; it remains exploratory.

See [`docs/LATE_STAGE_FINDINGS.md`](docs/LATE_STAGE_FINDINGS.md).

## Portable reproduction

One representative fresh-confirmatory training-time result now has a self-contained public reproduction path.

```bash
python -m pip install torch numpy scikit-learn
python scripts/reproduce/g7_confirmatory/run_seed.py --seed 4300 --out g7_seed_4300.json
```

In the recorded environment, this reproduced the archived G7 seed-4300 JSON exactly, with SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

See [`scripts/reproduce/g7_confirmatory/README.md`](scripts/reproduce/g7_confirmatory/README.md) and [`results/reproduction/g7_seed4300_report.json`](results/reproduction/g7_seed4300_report.json).

This validates software/data portability for an already-confirmatory seed; it is not counted as a new independent scientific replication.

## What is established

Within the tested small-model settings:

1. learned computation can exhibit task-conditioned compositional simplification;
2. simplification is not a simple local Canary-threshold phenomenon;
3. staged consolidation with intervening task learning can outperform direct contraction at the same final capacity;
4. recontracting can make the next compiler easier to optimize while simultaneously increasing downstream sensitivity to residual error;
5. task damage is better predicted by sensitivity-aware quantities than by representation error alone;
6. remaining learning horizon changes the expected post-commit damage trajectory.

## What is not established

- universal simplification of arbitrary neural networks;
- codec-independent Kolmogorov/MDL claims;
- large-pretrained-LLM validity;
- guaranteed wall-clock, FLOP, energy, RAM, or VRAM improvements;
- a universally optimal autonomous controller;
- that the current first/second-order risk model is architecture-universal.

## Potential applications

A longer-term systems interpretation is to treat a model as a **compact functional intermediate representation** rather than only as a tensor checkpoint:

```text
compact functional representation
        ↓
load / distribute / archive
        ↓
runtime compile or materialize
        ↓
hardware-specific executable computation
```

Possible directions include compact distribution, spanwise JIT materialization, native compact operators, cold-start reduction, edge deployment, hardware-specific recompilation, multi-model serving, checkpoint archival, and training-time self-recompilation.

These are **engineering hypotheses**, not established deployment wins.

See [`docs/APPLICATIONS.md`](docs/APPLICATIONS.md).

## Start here

1. [`docs/PUBLIC_SNAPSHOT.md`](docs/PUBLIC_SNAPSHOT.md) — reading order and snapshot policy.
2. [`docs/CORE_DISCOVERY.md`](docs/CORE_DISCOVERY.md) — the central empirical discovery and its scope.
3. [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md) — current supported/rejected/open claim registry.
4. [`docs/PUBLICATION_NOTES.md`](docs/PUBLICATION_NOTES.md) — publication-safe claim hierarchy.
5. [`docs/TRAINING_TIME_CONSOLIDATION.md`](docs/TRAINING_TIME_CONSOLIDATION.md) — G7–G17 mainline.
6. [`docs/LATE_STAGE_FINDINGS.md`](docs/LATE_STAGE_FINDINGS.md) — G18–G27 mechanism/controller results.
7. [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) — failed hypotheses retained as evidence.
8. [`docs/TERMINOLOGY.md`](docs/TERMINOLOGY.md) and [`docs/FAQ.md`](docs/FAQ.md) — definitions and interpretation boundaries.
9. [`docs/APPLICATIONS.md`](docs/APPLICATIONS.md) — possible deployment and systems applications.
10. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — evidence and reproduction policy.
11. [`docs/ROADMAP.md`](docs/ROADMAP.md) and [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md) — bounded conditional work and handoff.
12. [`STATUS.md`](STATUS.md) — current project state.

Machine-readable result summaries and protocol locks are under `results/`.

## Evidence classes

- **Confirmatory** — fresh seeds, locked condition/endpoints, explicit decision rule.
- **Independent holdout** — selected condition retested without reselection.
- **Exploratory** — implementation validation or hypothesis generation.
- **Negative result** — a failed hypothesis retained explicitly.
- **Reproduction** — rerun of an already-observed condition to validate portability; not a new independent seed by itself.

A successful toy result is not treated as universal evidence, and a failed local/post-hoc intervention is not treated as proof that training-time consolidation is impossible.

## Repository layout

- `src/canaria/` — cleaned reusable components from earlier phases.
- `scripts/phases/` — provenance-preserving evidence scripts; some historical scripts retain environment-specific paths.
- `scripts/reproduce/` — portable reproduction runners that remove historical local-path assumptions without rewriting evidence scripts.
- `results/` — machine-readable evidence, summary files, protocol locks, and reproduction reports.
- `docs/history/` and `docs/phases/` — historical research record.
- `archives/` — retained handoff/history material.
- `schemas/` — metadata schemas.

## License

Original code and documentation are released under the **Apache License 2.0**. Third-party datasets and libraries remain under their own licenses.

## Citation

See [`CITATION.cff`](CITATION.cff). Until a paper is published, cite the exact repository commit/snapshot plus the protocol/result files used.
