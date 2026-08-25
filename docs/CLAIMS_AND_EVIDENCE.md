# Claims and evidence

This file is the **current public claim registry** for Canaria. It separates supported empirical claims, rejected explanations, bounded engineering results, and open hypotheses. Detailed history remains in `RESEARCH_SUMMARY.md`, `GENERALIZATION_STATUS.md`, `TRAINING_TIME_CONSOLIDATION.md`, and the locked phase documents.

## A. Core discovery: compositional simplification

| Claim | Status | Main evidence | Scope / limitation |
|---|---|---|---|
| Learned computation can exhibit task-conditioned compositional simplification | **Supported in tested settings** | Original blinded/confirmatory span-composition program + fresh SmallViT + fresh residual-MLP direct replications | Operational replacement grammars; not codec-independent Kolmogorov complexity |
| Composition complexity is frequently subadditive | **Supported in original confirmatory setting** | `P(G>0)=0.7107`, 95% CI `0.6128–0.8137` | Residual-CNN-centered original setting and declared grammar |
| The core component-wise-versus-composed effect transfers to a Small Vision Transformer | **Supported** | 8/8 fresh seeds; mean composed/component-wise minimum-passing complexity ratio `0.51988`, bootstrap95 `[0.50634,0.53926]`; composed mean test utility `0.97856` | SmallViT on sklearn digits; fixed central two-block span and declared grammar |
| The core effect also transfers to a residual MLP under exact learned-parameter-budget matching | **Supported** | Fresh seeds `1200–1207`; component-wise mean minimum passing budget `3584`, composed `1728`; 8/8 lower; mean log2 ratio `-1.0519`, bootstrap95 `[-1.2075,-0.8962]`; test accuracy difference `+0.583 pt` | Residual MLP on sklearn digits; same broad supervised-classification genre; declared grammar |
| Much of the residual-MLP gap follows the composed functional objective rather than only one-module topology | **Supported as mechanistic secondary** | At 2048 params: local component NMSE `0.1474`, same two-module architecture jointly fit to span target `0.0639`, one composed module `0.0533` | One span/grammar; descriptive mechanistic secondary rather than universal law |
| High Canary is necessary for simplification | **Rejected** | Low-Canary strong-simplification rate `0.845`, 95% seed-cluster CI `0.7225–0.9500` | Other sensor definitions remain possible |
| Implementation-block boundaries are always the natural functional boundaries | **Rejected / unsupported** | Boundary expansion, wider-span replacements, residual-MLP joint span-objective control | Boundary behavior is architecture/task dependent |
| Local simplification is entirely hidden complexity relocation | **Rejected under measured codecs** | Whole-network accounting | Complexity remains codec dependent |
| Whole-network description size can be materially reduced in the residual-CNN setting | **Supported under declared codecs** | ~26.1% fixed-FP32 and ~28.8% q8+zlib reductions; exact 9,926-byte endpoint | Specific architecture/task and codec family |

See `CORE_DISCOVERY.md`, `CROSS_FAMILY_COMPOSITION_REPLICATION.md`, and `CORE_DISCOVERY_REPLICATION_DIGITS.md`.

## B. Training-time consolidation

| Claim | Status | Main evidence | Scope / limitation |
|---|---|---|---|
| Starting larger and consolidating during training can beat training the final small architecture from the start | **Supported** | G7 | Small real-text character LM |
| Correct function-aligned transfer matters at consolidation | **Supported** | G8 identity/shuffled controls | Tested transfer family |
| Perfect transfer fidelity is not required before task learning resumes | **Supported** | G9 dose response + G7 recovery | Does not imply arbitrary damage is recoverable |
| Structured inheritance alone explains the gain | **Rejected** | G10 | Inheritance helps when followed by functional refinement |
| Staged `4→3→2` consolidation with task learning can beat direct `4→2` | **Supported** | G15, fresh n=8 | Small real-text character LM |
| Merely factorizing one compiler fit into two explains the staged gain | **Rejected** | G17 equivalence control | Same testbed |
| The staged-path effect is specific to `4→3→2` | **Rejected by tested second family** | G19 `5→4→2` vs `5→2`, 8/8 wins, equal compiler updates | Two tested depth families, not universal |
| A calibration-only controller can autonomously reach the smaller target under a locked non-inferiority protocol | **Supported** | G11 | Current controller/testbed only |
| Remaining task-learning horizon should affect commit timing | **Supported** | G18 | Deadline-aware controller improved mean PPL and reduced mean compiler updates |

## C. Recontracting geometry and task sensitivity

| Claim | Status | Main evidence | Scope / limitation |
|---|---|---|---|
| After intermediate task learning, the next compiler can become easier to optimize in normalized functional-error terms | **Supported** | G20d: mean updates `41.5→32.125`, 8/8 seeds | Fixed standardized-error target; optimizer-update proxy |
| The same normalized compiler error becomes automatically more task-safe after recontracting | **Rejected** | G20e: immediate NLL damage increased | Same small LM testbed |
| Recontracting can simultaneously improve compiler fit conditioning and increase downstream sensitivity | **Supported** | G20d/e + G22 | Mechanistic interpretation remains local to tested models |
| Representation error magnitude alone is sufficient to predict task damage | **Rejected / incomplete** | G23 | Sensitivity-aware terms predict better |
| Error direction relative to task gradient improves immediate-damage prediction | **Supported** | G23 | Empirical local predictor, not a theorem |
| Adding a logit-space second-order term improves immediate-damage prediction | **Supported** | G24 | Same family; coefficients not claimed universal |
| The G24 fixed risk relation transfers without refitting to a different depth path | **Supported in tested transfer** | G25 `5→4→2`, 8/8 improvement | Same dataset/model family and head form |
| Remaining horizon improves future-damage prediction over treating immediate risk as constant | **Supported** | G26, fresh n=12, 12/12 improvement | Horizons 1/2/4 task epochs |

## D. Negative controller results

| Claim / intervention | Status | Evidence |
|---|---|---|
| A hard shadow task-damage veto is a generally better controller | **Rejected under tested protocol** | G21: only 10/12 reached final depth 2; compiler cost increased |
| One fixed future-risk cap automatically gives a cost/utility Pareto improvement | **Not supported** | G27 exploratory: strict cap bought utility with more fit; loose cap saved fit with worse utility |
| One static NMSE threshold is a complete notion of safe consolidation | **Not supported** | G18–G27 combined evidence |

## E. Autoregressive boundary results

| Claim | Status | Evidence | Limitation |
|---|---|---|---|
| Teacher-forced PPL is sufficient to certify autoregressive functional equivalence | **Rejected** | v22–v25 | Small causal models / short rollout metric |
| Natural-text post-hoc `4→2` compilation under the tested v23 budget preserves rollout behavior | **Rejected** | v23 | Aggressive fixed compression point |
| Teacher-forced logit KL is sufficient to repair that boundary | **Rejected** | v24 | One objective/weighting |
| One bounded on-policy trajectory-distillation iteration is sufficient | **Rejected** | v25 | One dataset-aggregation iteration |
| A substantially smaller compiler can preserve PPL while free-running trajectories diverge | **Supported** | v23–v25 | Character LM |

## F. Cross-architecture evidence

| Shift | Status | Headline |
|---|---|---|
| residual CNN → small ViT | **Adapted transfer supported** | >60% parameter reduction under tested adapted protocol |
| direct core phenomenon: residual-CNN evidence → SmallViT component-wise/composed test | **Fresh direct replication supported** | composed minimum-passing replacement complexity ~0.52× component-wise, 8/8 seeds |
| direct core phenomenon → residual-MLP component-wise/composed test | **Second fresh direct replication supported** | composed selected minimum budget geometric mean ~0.482× component-wise, 8/8 seeds; test utility noninferior and slightly higher in cohort |
| small ViT → non-image Transformer encoder | **Zero-shot transfer supported** | ~0.992 utility under tested sequence task |
| encoder/synthetic sequence → causal decoder | **Adapted transfer supported** | Repair restored PPL + generation utility |
| synthetic causal language → natural-English character LM | **No transfer under tested post-hoc budget** | PPL preserved while rollout fidelity failed |

The transfer map is intentionally mixed. See `GENERALIZATION_STATUS.md` for the detailed historical ledger.

## G. Public reproducibility closure

| Claim | Status | Evidence | Limitation |
|---|---|---|---|
| One representative confirmatory pipeline can run without private `/mnt/data` dependencies | **Supported** | Portable G7 seed-4300 runner | One already-confirmatory seed, not a new independent replication |
| The portable G7 seed-4300 output matches the archived output exactly in the recorded environment | **Supported** | Matching JSON SHA256 `68265c044f...d0028` | Exact match is environment-specific |

## H. Bounded runtime/materialization PoC

| Claim | Status | Evidence | Scope / limitation |
|---|---|---|---|
| A learned compact replacement can be serialized/materialized and executed directly without reconstructing the original larger model | **Supported in one small CPU PoC** | G7 seed-4300 runtime PoC | One model/seed and CPU environment |
| The compact serialized artifact is smaller | **Supported in PoC** | `110,093→54,646 B` (`−50.36%`) | `state_dict + manifest`; not a universal codec result |
| The compact model has lower measured CPU inference latency | **Supported in PoC** | batch-128 mean `47.05→23.11 ms`, 5 fresh-process probes | CPU-only and workload-specific |
| Load/materialization is universally faster | **Not established** | mean `7.85→5.86 ms`, but cache sensitivity observed | Secondary result only |
| Compact representation meaningfully reduces host RAM | **Not demonstrated** | RSS delta `4.72→4.56 MB` (`0.966×`) | Small model / allocator overhead dominates |
| GPU/LLM/energy/runtime benefits generalize | **Open** | No direct evidence | Requires dedicated systems benchmarks |

## I. Claims that remain open

- Universality across large pretrained Transformers or LLMs.
- Replication across additional task types, spans, widths, and replacement grammars.
- Codec-independent minimum description length.
- A universal mechanism dictionary or compiler grammar.
- General FLOP/energy/VRAM/RAM benefits of compact functional representations.
- GPU/NPU/edge/browser runtime effects.
- Whether spanwise JIT materialization reduces peak memory on realistically large models.
- A universally Pareto-optimal autonomous controller.
- Whether the local first/second-order task-risk relation generalizes across substantially different tasks, widths, heads, and architectures.

## Claim discipline

Do **not** describe the current evidence as proving that function composition always reduces mathematical complexity. The supported scientific statement is narrower:

> For some trained networks and task distributions, a composed input-output span admits a substantially smaller task-preserving replacement than component-wise treatment suggests. This effect was observed in the original residual-CNN program and directly replicated under locked fresh protocols in both a Small Vision Transformer and a residual MLP. Continued learning after consolidation can also change the ease and risk of later consolidation.

The supported systems statement is also narrow:

> In one small CPU PoC, a progressively consolidated learned representation was about 50% smaller as a serialized artifact and about 2× faster for the measured batch-128 inference workload, while meaningful host-RAM reduction was not demonstrated.
