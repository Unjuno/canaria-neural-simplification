# Claims and evidence

This file is the **current public claim registry** for Canaria. It intentionally separates supported empirical claims, rejected explanations, and open hypotheses. Detailed phase-by-phase history remains in `RESEARCH_SUMMARY.md`, `GENERALIZATION_STATUS.md`, `TRAINING_TIME_CONSOLIDATION.md`, and the locked phase documents.

## A. Core discovery: compositional simplification

| Claim | Status | Main evidence | Scope / limitation |
|---|---|---|---|
| Learned computation can exhibit task-conditioned compositional simplification | **Supported in tested settings** | Original blinded/confirmatory span-composition program | Operational replacement grammar; not codec-independent Kolmogorov complexity |
| Composition complexity is frequently subadditive | **Supported in original confirmatory setting** | `P(G>0)=0.7107`, 95% CI `0.6128–0.8137` | Residual-CNN-centered original setting and declared grammar |
| High Canary is necessary for simplification | **Rejected** | Low-Canary strong-simplification rate `0.845`, 95% seed-cluster CI `0.7225–0.9500` | Other sensor definitions remain possible |
| Implementation-block boundaries are always the natural functional boundaries | **Rejected / unsupported** | Boundary expansion and wider-span replacements | Boundary behavior is architecture/task dependent |
| Local simplification is entirely hidden complexity relocation | **Rejected under measured codecs** | Whole-network accounting | Complexity remains codec dependent |
| Whole-network description size can be materially reduced in the residual-CNN setting | **Supported under declared codecs** | ~26.1% fixed-FP32 and ~28.8% q8+zlib reductions; exact 9,926-byte endpoint | Specific architecture/task and codec family |

See `CORE_DISCOVERY.md` for the careful discovery statement.

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
| small ViT → non-image Transformer encoder | **Zero-shot transfer supported** | ~0.992 utility under tested sequence task |
| encoder/synthetic sequence → causal decoder | **Adapted transfer supported** | Repair restored PPL + generation utility |
| synthetic causal language → natural-English character LM | **No transfer under tested post-hoc budget** | PPL preserved while rollout fidelity failed |

The transfer map is intentionally mixed. See `GENERALIZATION_STATUS.md` for the detailed historical ledger.

## G. Claims that remain open

- Universality across large pretrained Transformers or LLMs.
- Codec-independent minimum description length.
- A universal mechanism dictionary or universal compiler grammar.
- Exact wall-clock/FLOP/energy/VRAM benefits of a compact functional representation.
- A universally Pareto-optimal autonomous controller.
- Whether the local first/second-order task-risk relation generalizes across substantially different tasks, widths, heads, and architectures.
- Whether compact functional representations can support practical runtime compilation/JIT materialization.

## Claim discipline

Do **not** describe the current evidence as proving that function composition always reduces mathematical complexity. The supported empirical statement is narrower:

> For some trained networks and task distributions, a composed input-output span admits a substantially smaller task-preserving replacement than component-wise treatment suggests, and continued learning after consolidation can change the ease and risk of later consolidation.
