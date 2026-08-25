# Canaria Neural Simplification

**Canaria** is an experimental research repository about a simple question:

> Can trained neural computation sometimes be represented more simply when several learned computations are treated as one composed function rather than as separate implementation blocks?

The repository preserves positive, negative, exploratory, confirmatory, reproduction, and bounded systems-PoC evidence. It is in a **research consolidation / public-snapshot phase**: broad experiment expansion is paused.

See [`docs/PUBLIC_SNAPSHOT.md`](docs/PUBLIC_SNAPSHOT.md) for the intended reading order and snapshot policy.

## Core empirical finding

The strongest project-level result is **task-conditioned compositional simplification of learned computation**.

In the tested settings:

- simplification was not confined to high-Canary regions;
- composition complexity was frequently subadditive (`P(G>0)=0.7107`, 95% CI `0.6128–0.8137`) under the original confirmatory grammar;
- implementation-block boundaries were not always the most natural functional boundaries;
- a learned span could sometimes admit a smaller task-conditioned replacement even when component-wise simplification was poor;
- whole-network accounting showed that measured local simplification was not merely hidden parameter relocation under the declared codecs.

A direct fresh replication now tests the core phenomenon in a different architecture family: a Small Vision Transformer on sklearn digits. For a fixed central two-block span, the same locked task/fidelity criterion was applied to component-wise versus directly composed replacement. Across 8/8 fresh eligible seeds, the selected composed representation used **4,904–5,424 replacement parameters** versus **9,808** for component-wise treatment. The mean composed/component-wise complexity ratio was **0.5199**, seed-bootstrap 95% CI **[0.5063, 0.5393]**, while mean held-out test utility of the selected composed candidates was **0.9786**.

The project does **not** claim that mathematical or Kolmogorov complexity universally decreases under composition. The supported claim is operational: for some trained networks and task distributions, a composed input-output map admits a substantially smaller task-preserving representation than component-wise treatment suggests. The direct ViT replication strengthens this beyond the original residual-CNN architecture, but does not establish universal Transformer or LLM behavior.

See [`docs/CORE_DISCOVERY.md`](docs/CORE_DISCOVERY.md) and [`docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md`](docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md).

## Dynamic extension: consolidation during learning

Later experiments showed that simplification is not only post-hoc.

A useful working process is:

> **form → transfer → commit → recontract → transfer again**

Headline evidence:

- **G7:** progressive `4→3→2` consolidation reached a final model with **52.28% fewer parameters** and beat early/late one-shot controls.
- **G15 + G17:** staged consolidation helped only when task learning occurred between commits; merely factorizing one compiler fit into two did not reproduce the benefit.
- **G18:** a deadline-aware controller improved mean PPL by **0.181** versus the static controller while reducing mean compiler updates from **184 to 136** on fresh `n=12`.
- **G19:** the staged effect replicated on `5→4→2` versus direct `5→2` by **−0.689 PPL** with identical compiler-update budgets, 8/8 fresh seeds.
- **G20d/G20e:** after recontracting, the next compiler reached the same normalized functional-error target with about **22% fewer updates**, but the same normalized error caused **more immediate task damage**.
- **G22–G26:** downstream sensitivity, error direction, logit-space second-order error, and remaining learning horizon materially improved task-damage prediction.
- **G21:** a hard shadow-damage veto failed because it prevented final contraction in 2/12 fresh seeds.
- **G27:** fixed risk caps exposed a cost/utility trade-off rather than a Pareto improvement; it remains exploratory.

See [`docs/TRAINING_TIME_CONSOLIDATION.md`](docs/TRAINING_TIME_CONSOLIDATION.md) and [`docs/LATE_STAGE_FINDINGS.md`](docs/LATE_STAGE_FINDINGS.md).

## Portable exact reproduction

A self-contained public runner reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` imports.

```bash
python -m pip install -r scripts/reproduce/g7_confirmatory/requirements.txt
python scripts/reproduce/g7_confirmatory/run_seed.py --seed 4300 --out g7_seed_4300.json
```

In the recorded environment, the complete JSON exactly matched the archived confirmatory output with SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

See [`scripts/reproduce/g7_confirmatory/README.md`](scripts/reproduce/g7_confirmatory/README.md) and [`results/reproduction/g7_seed4300_report.json`](results/reproduction/g7_seed4300_report.json).

This is portability/reproducibility evidence for an already-confirmatory seed, not a new independent scientific replication.

## Bounded runtime/materialization PoC

A small CPU-only proof of concept tests the deployment idea:

```text
compact learned representation
→ serialize
→ load/materialize
→ execute directly
```

Using G7 seed 4300:

| metric | large 4-block | progressive compact 2-block |
|---|---:|---:|
| serialized artifact + manifest | 110,093 B | **54,646 B** |
| parameters | 23,138 | **11,042** |
| batch-128 CPU inference, 5 fresh-process probes | 47.05 ms | **23.11 ms** |
| load/materialize, mean | 7.85 ms | 5.86 ms |
| process RSS delta | 4.72 MB | 4.56 MB |
| test PPL | 19.2784 | **18.9322** |

Interpretation:

- storage/distribution reduction was demonstrated in this small PoC;
- direct CPU execution was faster in this measured setup;
- load/materialization was faster on average but cache-sensitive, so it is secondary evidence;
- meaningful host-RAM reduction was **not demonstrated**;
- GPU, large-model, energy, and universal runtime claims are not established.

See [`docs/RUNTIME_POC.md`](docs/RUNTIME_POC.md) and [`results/reproduction/runtime_poc_seed4300_report.json`](results/reproduction/runtime_poc_seed4300_report.json).

## What is established

Within the tested small-model settings:

1. learned computation can exhibit task-conditioned compositional simplification;
2. a direct component-wise-versus-composed replication in a Small Vision Transformer found about **48% lower replacement complexity** for the composed span under locked task/fidelity criteria, with 8/8 fresh seeds favoring composition;
3. simplification is not a simple local Canary-threshold phenomenon;
4. staged consolidation with intervening task learning can outperform direct contraction at the same final capacity;
5. recontracting can make the next compiler easier to optimize while simultaneously increasing downstream sensitivity to residual error;
6. task damage is better predicted by sensitivity-aware quantities than by representation error alone;
7. remaining learning horizon changes expected post-commit damage;
8. one representative confirmatory path is publicly reproducible without private local imports;
9. one small CPU PoC demonstrates that a learned compact representation can be serialized, materialized, and executed directly without reconstructing the original larger model.

## What is not established

- universal simplification of arbitrary neural networks;
- codec-independent Kolmogorov/MDL claims;
- large-pretrained-LLM validity;
- universal Transformer-span subadditivity;
- guaranteed FLOP, wall-clock, energy, RAM, or VRAM improvements;
- universal GPU/runtime speedup;
- a universally optimal autonomous controller;
- architecture-universal risk-model coefficients.

## Applications

The longer-term systems interpretation is to treat a trained model as a **compact functional intermediate representation** rather than only as a tensor checkpoint.

Potential directions include compact distribution, spanwise JIT materialization, native compact operators, cold-start reduction, edge deployment, hardware-specific recompilation, multi-model serving, checkpoint archival, and training-time self-recompilation.

The small runtime PoC provides initial evidence for compact serialization and direct CPU execution only. The broader application directions remain engineering hypotheses until measured at the relevant scale/resource.

See [`docs/APPLICATIONS.md`](docs/APPLICATIONS.md).

## Start here

1. [`docs/PUBLIC_SNAPSHOT.md`](docs/PUBLIC_SNAPSHOT.md) — reading order and snapshot policy.
2. [`docs/CORE_DISCOVERY.md`](docs/CORE_DISCOVERY.md) — central empirical discovery and scope.
3. [`docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md`](docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md) — direct SmallViT component-wise versus composed replication.
4. [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md) — supported/rejected/open claim registry.
5. [`docs/PUBLICATION_NOTES.md`](docs/PUBLICATION_NOTES.md) — publication-safe claim hierarchy.
6. [`docs/TRAINING_TIME_CONSOLIDATION.md`](docs/TRAINING_TIME_CONSOLIDATION.md) — G7–G17 mainline.
7. [`docs/LATE_STAGE_FINDINGS.md`](docs/LATE_STAGE_FINDINGS.md) — G18–G27 mechanisms/controllers.
8. [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) — failed hypotheses retained as evidence.
9. [`docs/RUNTIME_POC.md`](docs/RUNTIME_POC.md) — bounded systems result.
10. [`docs/TERMINOLOGY.md`](docs/TERMINOLOGY.md) and [`docs/FAQ.md`](docs/FAQ.md) — definitions and interpretation boundaries.
11. [`docs/APPLICATIONS.md`](docs/APPLICATIONS.md) — deployment directions and evidence status.
12. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — reproduction policy.
13. [`docs/ROADMAP.md`](docs/ROADMAP.md) and [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md) — handoff and future work.
14. [`STATUS.md`](STATUS.md) — current project state.

## Evidence classes

- **Confirmatory** — fresh seeds, locked condition/endpoints, explicit decision rule.
- **Independent holdout** — selected condition retested without reselection.
- **Exploratory** — implementation validation or hypothesis generation.
- **Negative result** — a failed hypothesis retained explicitly.
- **Reproduction** — rerun of an already-observed condition to validate portability; not a new independent seed by itself.
- **Systems PoC** — bounded engineering measurement; scope is limited to the measured environment/resource.

## Repository layout

- `src/canaria/` — cleaned reusable components from earlier phases.
- `scripts/phases/` — provenance-preserving evidence scripts; some retain historical environment-specific paths.
- `scripts/reproduce/` — portable reproduction and systems-PoC runners.
- `scripts/replication/` — fresh direct replication runners for the core scientific phenomenon.
- `results/` — machine-readable evidence, protocol locks, reproduction reports, replication results, and PoC reports.
- `docs/history/` and `docs/phases/` — historical research record.
- `archives/` — retained handoff/history material.
- `schemas/` — metadata schemas.

## License

Original code and documentation are released under the **Apache License 2.0**. Third-party datasets and libraries remain under their own licenses.

## Citation

See [`CITATION.cff`](CITATION.cff). Until a paper is published, cite the exact repository commit/snapshot plus the protocol/result files used.
