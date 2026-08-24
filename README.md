# Canaria Neural Simplification

**Canaria** is an experimental research repository for studying task-conditioned computational simplification, redistribution, compilation, and low-bit compression in trained neural networks.

> **Project status (2026-08-24): active external-validity/generalization experiments.**

The repository is designed as an auditable research record: confirmatory results, pilot/adaptation history, negative results, exact codecs, machine-readable summaries, and historical failures are kept together rather than publishing only the successful endpoint.

## Current strongest findings

- **Simplification is not confined to high-Canary regions.** In blinded Phase A, the low-Canary strong-simplification rate was **0.845** (95% seed-cluster CI **0.7225–0.9500**). A Canary-local necessary-condition hypothesis failed.
- **Composition subadditivity is common in the original tested setting.** Confirmatory `P(G>0)` was **0.7107** (95% CI **0.6128–0.8137**).
- **Pure downstream-location explanations failed.** Equal-capacity pre/post interventions did not support a special post-boundary causal advantage.
- **Whole-network simplification survives accounting in the residual-CNN setting.** Compiled models were about **26.1% smaller in fixed FP32 code** and **28.8% smaller under q8+zlib** versus matched controls while retaining utility.
- **A real whole-network codec below 10 KB was independently confirmed.** The current residual-CNN endpoint is **9,926 bytes**, exact pack/unpack, with confirmatory combined fidelity **0.9636 [0.9516, 0.9740]** and matched-control utility **0.9835 [0.9639, 1.0059]**.
- **Transformer generalization is mixed, not uniformly positive.** The same broad simplification idea transfers to a small ViT (adapted), a non-image Transformer encoder (zero-shot), and a synthetic causal decoder (adapted), but **fails under the tested budget on a held-out natural-English character LM**.
- **Teacher-forced PPL is not sufficient for autoregressive simplification.** In v23, tau0 PPL utility was **0.9970 [0.9958, 0.9982]** after a 52.28% parameter reduction, yet greedy rollout agreement was only **0.6326 [0.5503, 0.7269]**. The prespecified tau8 joint-repair adaptation also failed.

These are **task/architecture/codec-conditioned empirical results**, not a universal theorem or a proof of codec-independent minimum description length.

## Generalization map

| Phase | Shift | Outcome | Headline |
|---|---|---|---|
| **G3 / v20** | residual CNN -> small ViT | **A — adapted transfer** | 4 Transformer blocks -> 2 smaller blocks; 60.18% parameter reduction; tau8 utility 0.9685 [0.9610, 0.9767] |
| **G5 / v21** | image-token ViT -> non-image Transformer encoder | **Z — zero-shot transfer** | tau0 utility 0.99184 [0.97986, 1.00199]; ~58% q8+zlib state-stream reduction |
| **G6 / v22** | encoder/synthetic sequence -> causal decoder | **A — adapted transfer** | zero-shot PPL looked acceptable but generation drifted; tau8 restored PPL + generation utility |
| **G6b / v23** | synthetic causal language -> held-out natural English character LM | **N — no transfer under tested budget** | tau0 PPL nearly preserved but autoregressive trajectory fidelity failed; bounded joint repair also failed |

The project therefore distinguishes three different questions:

1. **Phenomenon universality** — does task-conditioned simplification recur across network families?
2. **Compiler universality** — does one unchanged compiler work everywhere?
3. **Adaptation-rule universality** — can a small explicit set of family/task-specific rules predict when simplification is recoverable?

The present evidence supports a **conditional/mixed** picture rather than universal zero-shot transfer.

## Start here

1. [`docs/GENERALIZATION_STATUS.md`](docs/GENERALIZATION_STATUS.md) — live cross-architecture transfer ledger.
2. [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md) — claim-by-claim status, evidence class, and limitation.
3. [`docs/KNOWLEDGE_MAP.md`](docs/KNOWLEDGE_MAP.md) — question → evidence navigation.
4. [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md) — integrated research narrative through the core/global-accounting program.
5. [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) — failed explanations retained as first-class evidence.
6. [`docs/GENERALIZATION_ROADMAP.md`](docs/GENERALIZATION_ROADMAP.md) — Z/A/C/N/I transfer taxonomy and future architecture/task sequence.
7. [`docs/phases/README.md`](docs/phases/README.md) — chronological protocol/result index.
8. [`results/README.md`](results/README.md) — machine-readable evidence index.
9. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — seed policy, matched controls, blindness, storage terminology, metadata rules.
10. [`STATUS.md`](STATUS.md) — current execution state and repository policy.

## Recent generalization phases

### v20 — small ViT
- [`docs/phases/v20/63_G3_SMALL_VIT_GENERALIZATION_PROTOCOL_V20.md`](docs/phases/v20/63_G3_SMALL_VIT_GENERALIZATION_PROTOCOL_V20.md)
- [`docs/phases/v20/64_G3_SMALL_VIT_GENERALIZATION_RESULTS_V20.md`](docs/phases/v20/64_G3_SMALL_VIT_GENERALIZATION_RESULTS_V20.md)

### v21 — non-image Transformer encoder
- [`docs/phases/v21/66_G5_SEQUENCE_TRANSFORMER_CONFIRM_PROTOCOL_V21.md`](docs/phases/v21/66_G5_SEQUENCE_TRANSFORMER_CONFIRM_PROTOCOL_V21.md)
- [`docs/phases/v21/67_G5_SEQUENCE_TRANSFORMER_RESULTS_V21.md`](docs/phases/v21/67_G5_SEQUENCE_TRANSFORMER_RESULTS_V21.md)

### v22 — synthetic causal decoder LM
- [`docs/phases/v22/69_G6_DECODER_LM_CONFIRM_PROTOCOL_V22.md`](docs/phases/v22/69_G6_DECODER_LM_CONFIRM_PROTOCOL_V22.md)
- [`docs/phases/v22/70_G6_DECODER_LM_RESULTS_V22.md`](docs/phases/v22/70_G6_DECODER_LM_RESULTS_V22.md)

### v23 — held-out natural-English character LM (negative transfer)
- [`docs/phases/v23/71_75_G6B_PILOT_AUDIT_V23.md`](docs/phases/v23/71_75_G6B_PILOT_AUDIT_V23.md)
- [`docs/phases/v23/76_G6B_REALTEXT_CONFIRM_PROTOCOL_V23.md`](docs/phases/v23/76_G6B_REALTEXT_CONFIRM_PROTOCOL_V23.md)
- [`docs/phases/v23/77_G6B_REALTEXT_LM_RESULTS_V23.md`](docs/phases/v23/77_G6B_REALTEXT_LM_RESULTS_V23.md)
- [`results/v23/g6b_confirmatory_summary.json`](results/v23/g6b_confirmatory_summary.json)

## Core / whole-network compression milestones

- 44.5-byte compiled core independently confirmed after repair (v17).
- Whole-network accounting showed local simplification was not explained away by measured shell-code growth (v18).
- Exact independently confirmed **9,926-byte whole-network binary model** in the residual-8 digits setting (v19).

Important: **44.5 B / ~28 B refer to compiled-core representations, not complete networks. 9,926 B is the current exact standalone whole-network serialized result.**

## Evidence classes

- **Confirmatory** — condition/protocol locked before outcome inspection; independent seeds and explicit decision rule.
- **Independent holdout** — a previously selected condition retested on fresh seeds/checkpoints without reselection.
- **Pilot / exploratory** — used for bounded selection, implementation validation, or hypothesis generation.
- **Negative result** — a failed hypothesis or transfer condition retained explicitly rather than discarded.

For autoregressive models, PPL/token accuracy and free-running rollout are reported separately. A PPL-only success is not accepted as functional transfer after v22/v23.

## Current next questions

The highest-information follow-ups are:

- measure v23 rollout divergence versus horizon (1/2/4/8/16/24 tokens);
- test a **logit/KL-aware compiler objective** against residual-stream MSE on pilots, then freeze and independently confirm;
- test a small **pretrained/subword real-text LM** using independent fine-tuning/checkpoint replicates;
- execute CIFAR-10 ResNet/ViT branches to separate language/autoregression from general task difficulty;
- continue task-effective repair-dimension, off-manifold, null-model, mechanism-algebra, and codec-independent MDL work.

See [`docs/GENERALIZATION_STATUS.md`](docs/GENERALIZATION_STATUS.md) and [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md).

## Reuse / code organization

- `src/canaria/` — cleaned reusable components.
- `scripts/phases/` — evidence-producing historical/phase scripts; portability fixes should be additive rather than silently rewriting provenance.
- `results/` — compact machine-readable evidence.
- `docs/phases/` — locked protocols, amendments, and result narratives.
- `schemas/` — run/blind-event metadata schemas.

## Quick environment check

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/audit_repo.py
```

Early historical runs did not preserve a single exact package lockfile. Read [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and [`environment/history/v10/REPRODUCIBILITY_LIMITS.md`](environment/history/v10/REPRODUCIBILITY_LIMITS.md) before expecting bitwise reproduction.

## License

Original code and documentation are released under the **Apache License 2.0**. Third-party datasets/libraries remain under their own licenses.

## Citation

See [`CITATION.cff`](CITATION.cff). Until a paper is published, cite the repository commit/snapshot and the exact protocol/result files used.
