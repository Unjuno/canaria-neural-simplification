# Canaria Neural Simplification

**Canaria** is an experimental research repository for studying task-conditioned computational simplification, functional consolidation, self-recompilation during learning, and low-description-length representations of trained neural computation.

> **Project status (2026-08-24): active training-time consolidation / autonomous-controller experiments.**

The repository is an auditable research record. Confirmatory results, exploratory adaptations, negative results, protocol locks, and historical failures are kept together rather than publishing only successful endpoints.

## Current working thesis

The current mainline no longer treats Canaria as only post-hoc pruning or one-shot compression. The working process is:

> **form → transfer → commit → recontract → transfer again**

A larger core first forms useful computation. A learned span is then transferred into a smaller replacement. The replacement is committed before perfect equivalence is required, and task learning resumes so the surrounding network can reorganize around the new mechanism. Later consolidations repeat the process.

This training-time view was adopted after the v23–v25 natural-text post-hoc series exposed a limitation of teacher-forced fidelity: local/PPL preservation could coexist with poor autoregressive trajectory agreement. Those negative results remain valid and are preserved as part of the evidence chain.

## Current strongest findings

### Historical / cross-architecture program

- **Simplification is not confined to high-Canary regions.** In blinded Phase A, the low-Canary strong-simplification rate was **0.845** (95% seed-cluster CI **0.7225–0.9500**).
- **Composition subadditivity was common in the original tested setting.** Confirmatory `P(G>0)` was **0.7107** (95% CI **0.6128–0.8137**).
- **Whole-network simplification survived accounting in the residual-CNN setting.** Compiled models were about **26.1% smaller in fixed FP32 code** and **28.8% smaller under q8+zlib** versus matched controls while retaining utility.
- A real exact whole-network codec below 10 KB was independently confirmed: **9,926 bytes** in the residual-CNN endpoint.
- Transformer/generalization results were mixed. Small ViT, a non-image Transformer encoder, and a synthetic causal decoder transferred under different adaptation regimes, while the tested post-hoc natural-English character-LM protocol failed rollout-sensitive criteria.
- **Teacher-forced PPL is not sufficient evidence of autoregressive functional equivalence.** The v23–v25 negative series is retained as a boundary result, not deleted by the later training-time work.

### Corrected training-time mainline: G7–G17

All headline results below use the small real-text character-LM testbed inherited from v23–v25. They are mechanistic results, not yet evidence for large pretrained LMs.

- **G7 — progressive training-time consolidation:** final architecture `4 blocks / MLP48 → 3/36 → 2/24`, a **52.28% parameter reduction**. Fresh seeds 4300–4307. Progressive consolidation beat early one-shot by **−0.304 PPL** (95% CI **[−0.338, −0.266]**) and late one-shot by **−0.420 PPL** (**[−0.511, −0.323]**), 8/8 seeds in both comparisons. Training the final small model from the start was substantially worse.
- **G8 — correct functional handoff matters:** with the architecture schedule fixed, fitting the replacement to the true span function beat identity and shuffled-target controls by **−9.967 PPL** and **−3.018 PPL**, respectively, 8/8 fresh seeds.
- **G9 — transfer accuracy has a dose response with diminishing returns:** 25→50%, 50→100%, and 100→200% compiler-fit budgets improved final PPL by about **0.501**, **0.236**, and **0.097**, respectively.
- **G10 — structured inheritance helps only when followed by functional refinement:** inheritance alone was **+1.223 PPL** worse than functional fitting; inheritance followed by functional refinement was **−0.122 PPL** better than random-init functional fitting, 8/8 seeds.
- **G11 — autonomous consolidation:** a calibration-only controller autonomously reached the final 2-block model in **8/8** fresh seeds. Mean Auto PPL **20.1767** versus Large **20.2267**; paired relative difference **−0.240%**, 95% CI **[−0.593%, +0.153%]**, satisfying the preregistered +2% non-inferiority criterion. Mean compiler updates: **180**, maximum **192**.
- **G15 — staged consolidation beats waiting for one large merge:** `4→3→2` with task learning between commits beat a direct/wait `4→2` path by **−0.2993 PPL**, 95% CI **[−0.3379, −0.2616]**, **8/8** fresh seeds, with comparable or slightly lower compiler effort.
- **G17 — two-step fitting alone does not explain G15:** when `4→3→2` was performed immediately with **no task learning between the two compiler fits**, its final PPL was equivalent to direct `4→2`. Mean factorized−direct difference **+0.0279 PPL**, 95% CI **[−0.0019, +0.0614]**, entirely inside the preregistered equivalence band **[−0.10, +0.10]**.

The combined G15+G17 result is the strongest current mechanistic evidence: **the benefit is associated with learning/recontracting between consolidation events, not merely with algebraically factorizing one compression into two fits.**

## What is established vs not established

### Supported within the current small real-text LM testbed

1. Starting large and later consolidating can outperform training the final small architecture from the start.
2. Correct function-aligned transfer at a consolidation event is materially important.
3. Perfect zero-error transfer is not required; continued task learning repairs substantial consolidation damage.
4. A staged path with task learning between commits can outperform a direct merge at the same final capacity.
5. The staged advantage is not reproduced by merely splitting one compiler fit into two fits without intervening task learning.
6. A calibration-only controller can make consolidation decisions without test-set access and reach the target smaller architecture under a locked non-inferiority protocol.

### Not established

- That the mechanism generalizes to large pretrained Transformers.
- That MSE is the optimal transfer objective.
- That the current compiler-cost proxy is equivalent to exact FLOPs, energy, or wall-clock cost.
- That the internal cause of recontracting is uniquely identified. Architecture regularization, basin migration, representation redistribution, and true reduction in functional description length remain competing explanations.
- That one static NMSE threshold is an optimal autonomous policy.

## Start here

1. [`docs/TRAINING_TIME_CONSOLIDATION.md`](docs/TRAINING_TIME_CONSOLIDATION.md) — corrected mainline, G7–G17 evidence, interpretation, and limitations.
2. [`docs/NEXT_EXPERIMENTS_AUTONOMOUS.md`](docs/NEXT_EXPERIMENTS_AUTONOMOUS.md) — G18/G19 next tests.
3. [`results/training_time/summary.json`](results/training_time/summary.json) — machine-readable headline results.
4. [`results/training_time/protocol_manifest.json`](results/training_time/protocol_manifest.json) — seed ranges, preregistered decisions, and protocol SHA256 values.
5. [`scripts/phases/training_time/stable_auto_controller_v2.py`](scripts/phases/training_time/stable_auto_controller_v2.py) — current autonomous-controller evidence script.
6. [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md) — historical claim registry.
7. [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) — failed explanations and transfer conditions retained as first-class evidence.
8. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — repository-wide reproducibility policy.
9. [`STATUS.md`](STATUS.md) — current execution state.

## Generalization history

| Phase | Shift | Outcome | Headline |
|---|---|---|---|
| G3 / v20 | residual CNN → small ViT | adapted transfer | 4 Transformer blocks → 2 smaller blocks; 60.18% parameter reduction |
| G5 / v21 | image-token ViT → non-image Transformer encoder | zero-shot transfer | tau0 utility ~0.992; ~58% q8+zlib state-stream reduction |
| G6 / v22 | encoder/synthetic sequence → causal decoder | adapted transfer | bounded repair restored PPL + generation utility |
| G6b / v23 | synthetic causal language → held-out natural English character LM | negative under tested post-hoc budget | PPL nearly preserved while autoregressive trajectory fidelity failed |
| G7–G17 / 2026-08-24 | post-hoc simplification → training-time consolidation | positive mechanistic program | functional handoff + recontracting + autonomous staged consolidation |

The project therefore distinguishes at least four questions:

1. **Phenomenon universality** — does task-conditioned simplification recur across network families?
2. **Compiler universality** — does one unchanged compiler work everywhere?
3. **Adaptation-rule universality** — can small explicit rules predict recoverable simplification?
4. **Training-path simplification** — can a learning system repeatedly recompile its own computation while retaining or improving task utility?

## Evidence classes

- **Confirmatory** — protocol/condition locked before outcome inspection; fresh seeds and explicit decision rule.
- **Independent holdout** — selected condition retested without reselection.
- **Pilot / exploratory** — implementation validation, bounded selection, or hypothesis generation.
- **Negative result** — failed hypothesis or transfer condition retained explicitly.

A failed local/post-hoc transfer is not treated as proof that a training-time consolidation path is impossible, and a successful toy training-time result is not treated as proof of universal simplifiability.

## Code organization

- `src/canaria/` — cleaned reusable components from earlier phases.
- `scripts/phases/` — evidence-producing historical scripts; provenance-preserving scripts may contain environment-specific paths and should not be silently rewritten.
- `scripts/phases/training_time/` — current G11/G15/G17 controller and mechanistic controls.
- `results/` — compact machine-readable evidence.
- `docs/phases/` — historical locked protocols and results.
- `docs/TRAINING_TIME_CONSOLIDATION.md` — current corrected mainline.
- `schemas/` — run/blind-event metadata schemas.

## Quick environment check

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/audit_repo.py
```

Historical evidence-producing scripts may depend on paths/modules preserved in their corresponding archive. Read [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before expecting bitwise reproduction from an isolated script.

## License

Original code and documentation are released under the **Apache License 2.0**. Third-party datasets/libraries remain under their own licenses.

## Citation

See [`CITATION.cff`](CITATION.cff). Until a paper is published, cite the repository commit/snapshot and the exact protocol/result files used.
