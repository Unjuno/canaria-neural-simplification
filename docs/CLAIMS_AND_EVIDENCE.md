# Claims and evidence

This is the **reviewed baseline claim registry** for Canaria after the 2026-08-26 independent re-review. It is not yet the final broad-announcement claim set; current readiness and later-evidence inclusion decisions are tracked in `ANNOUNCEMENT_READINESS.md` / Issue #13.

For the decision ledger, see `INDEPENDENT_REREVIEW_2026-08-26.md`. Historical documents remain available, but when wording conflicts with this registry, this file defines the reviewed baseline until a later explicit scientific review changes it.

## A. Core discovery — operational compositional simplification

| Claim | Baseline status | Main evidence | Scope / limitation |
|---|---|---|---|
| Some learned spans admit a smaller task-preserving replacement when fitted as one composed input-output function than when simplified at implementation-component boundaries | **KEEP — supported in tested settings** | Original residual-CNN program + fresh SmallViT + fresh residual-MLP direct replications | Operational replacement/description complexity under declared task distributions, grammars, criteria, and accounting; not Kolmogorov complexity |
| Positive composition gain was frequent under the original declared grammar | **KEEP after wording edit** | Original confirmatory `P(G>0)=0.7107`, 95% CI `0.6128–0.8137` | Do not restate as an intrinsic mathematical law that “composition complexity is subadditive” |
| The direct effect transfers to a Small Vision Transformer | **KEEP with isolation caveat** | 8/8 fresh eligible seeds; mean selected composed/component-wise replacement-parameter ratio `0.51988`, bootstrap95 `[0.50634,0.53926]`; composed mean test utility `0.97856` | SmallViT on sklearn digits, one fixed two-block span and declared grammar. The locked selection rule excludes test metrics, but the runner records test metrics for every candidate; test was therefore not operationally hidden during result generation |
| The direct effect transfers to a residual MLP under exact learned replacement-parameter matching | **KEEP** | Fresh `1200–1207`; component-wise mean minimum passing budget `3584`, composed `1728`; 8/8 lower; mean log2 ratio `-1.0519`, bootstrap95 `[-1.2075,-0.8962]` | Residual MLP on sklearn digits; validation-only budget selection; declared grammar |
| The residual-MLP joint-factorized control recovers most of the local component-wise NMSE gap | **KEEP as descriptive/mechanistic secondary** | At 2048 params: local `0.1474`, joint span-objective `0.0639`, single composed `0.0533` | No confirmatory pass rule. Consistent with much of the gap following the composed span objective; not a causal decomposition theorem |
| High Canary is necessary for simplification | **REJECTED** | Low-Canary strong-simplification rate `0.845`, 95% seed-cluster CI `0.7225–0.9500` | Other sensors remain possible |
| Implementation-block boundaries are always the natural functional boundaries | **REJECTED / unsupported** | Boundary expansion, wider-span replacement, direct component-wise/composed controls | Architecture/task dependent |
| Local simplification is entirely hidden complexity relocation | **REJECTED under measured codecs** | Whole-network accounting | Codec dependent; not a universal minimum-description result |

See `CORE_DISCOVERY.md`, `CROSS_FAMILY_COMPOSITION_REPLICATION.md`, and `CORE_DISCOVERY_REPLICATION_DIGITS.md`.

Candidate post-snapshot research, including draft regression external-validity work, is **not automatically added to this table**. It requires a separate inclusion/exclusion review.

## B. Training-time consolidation

The training-time evidence is a separate extension on a **small real-text character-LM testbed**. Compiler cost is generally an optimizer-update or replacement-parameter × update proxy, not measured FLOPs, energy, or wall-clock cost.

| Claim | Baseline status | Evidence / boundary |
|---|---|---|
| Progressive G7 consolidation beat the preregistered early and late one-shot controls | **KEEP — confirmatory primary** | Progressive − early `-0.3037` PPL, bootstrap95 `[-0.3384,-0.2664]`; progressive − late `-0.4197`, `[-0.5114,-0.3229]` |
| G7 progressive also outperformed the small-from-start condition in that cohort | **KEEP as secondary observation** | Small-from-start mean PPL `21.3126` versus progressive `19.4478`; this was not the G7 primary PASS comparison |
| Function-aligned transfer matters under the tested controls | **KEEP** | G8 functional versus identity/shuffled controls |
| Staged consolidation with intervening task learning can outperform direct contraction under the tested schedules | **KEEP** | G15 `4→3→2` versus direct-wait `4→2`; G19 `5→4→2` versus `5→2` |
| Merely splitting one compiler fit into two explains the staged gain | **REJECTED** | G17 back-to-back factorized fitting was equivalent to direct `4→2` within the preregistered band |
| The tested deadline-aware controller improved over the tested static NMSE controller | **KEEP, specifically scoped** | G18: mean PPL `19.9384→19.7574`, paired difference `-0.1810`, bootstrap95 `[-0.3508,-0.0417]`; mean compiler updates `184→136` | Do not generalize to a universal rule that remaining horizon must determine all commit timing |
| Recontracting can make the next compiler reach a fixed normalized error target with fewer updates while residual error becomes more task-damaging | **KEEP within testbed** | G20d/e, G22 | “Easier to fit” is not “safer to approximate” |
| Error direction / logit-space terms / remaining horizon improved task-damage prediction in the tested protocols | **KEEP within testbed** | G23–G26 | Empirical predictors; coefficients are not mathematical laws |
| A hard shadow-damage veto is generally better | **REJECTED under tested protocol** | G21 failed target reach in 2/12 and increased compiler updates |
| One fixed future-risk cap yields a cost/utility Pareto improvement | **NOT SUPPORTED** | G27 exploratory only |

## C. Phase 2 — precision, quantization, and repair

The authoritative correction registry is `../results/phase2/precision_composition/CORRECTION_STATUS.json`.

| Phase / claim | Baseline status | Boundary |
|---|---|---|
| 2A: 4-bit composed minimum passing coded size is lower | **KEEP — VALID_PASS** | Locked residual-MLP experiment; declared symmetric signed-uniform quantizer and FP16 scale metadata |
| 2B: increasing weight count alone rescues naive 3-bit per-matrix PTQ | **REJECTED — VALID_FAIL** | No rescue through 16,384 weights in the tested fresh cohort |
| 2C: row-wise scales can rescue 3-bit PTQ | **KEEP — VALID_PASS** | 7/8 passes for both topologies; rescue is not uniquely compositional |
| 2E: stochastic repair failure / 0-of-8 composed result | **INVALIDATE** | `INVALIDATED_IMPLEMENTATION_BUG`: repair used raw `Xt` instead of internal activation `ta[0]`; equal width 64 hid the semantic error. `DO_NOT_USE_FOR_INFERENCE` |
| 2I: repair RNG explains 2E | **REMOVE / causal claim retracted** | Activation domain changed as well as RNG |
| 2H/2J explanations tied to 2E | **EDIT** | Numerical observations may remain, but bug-defined cohort/mechanism comparisons are weakened or confounded |
| Correct activation-domain short QAT-style repair can make coarse per-matrix 3-bit viable in this residual-MLP family | **KEEP with provenance boundary** | Corrected 2D/2L-family evidence. Later correction archive SHA256 `1a339be12d7644de534ac77a712307c49ee0c3d9acb28c8a3532883edca3dab7`; not all later raw per-seed artifacts are checked into this Git branch |
| Composition has lower QAT repair sample complexity | **REMOVE as positive claim** | Phase 2O is `VALID_UNCERTAIN`: exact sign `p=0.1662`; bootstrap95 mean difference `[-157.1,+58.2]` samples |

Invalidation history is preserved in `../results/phase2/precision_composition/INVALIDATED_HISTORY.md`; it is not silently deleted.

## D. Reproducibility and systems boundary

| Claim | Baseline status | Evidence / limitation |
|---|---|---|
| One representative G7 confirmatory pipeline runs without private `/mnt/data` dependencies | **KEEP** | Portable seed-4300 runner |
| The portable seed-4300 output exactly matches the archived output in the recorded environment | **KEEP** | Matching JSON SHA256 `68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028` | Software/portability reproduction of an already-confirmatory seed; **not** a new independent scientific replication |
| A compact learned G7 artifact can be serialized, materialized, and executed directly without reconstructing the larger model | **KEEP in one small CPU PoC** | G7 seed 4300 runtime PoC |
| The compact serialized artifact is smaller | **KEEP in PoC** | `110,093→54,646 B` (`-50.36%`) |
| The compact model has lower measured CPU batch-128 inference latency | **KEEP in PoC** | Mean `47.05→23.11 ms` over five fresh-process probes |
| Meaningful host-RAM reduction was demonstrated | **REJECTED / not demonstrated** | RSS delta `4.72→4.56 MB`; allocator/process overhead dominates at this scale |
| GPU/VRAM/energy/large-model/general runtime speedup is established | **OPEN / not established** | No direct evidence |

The pre-announcement gate additionally requires a pinned-environment reproduction of the full residual-MLP `1200–1207` cohort. That rerun remains reproduction evidence and does not change the original confirmatory seed count.

## E. Claims that remain open

- Universality across large pretrained Transformers or LLMs.
- Replication across substantially different tasks, spans, widths, and replacement grammars.
- Strong-teacher regression external validity suitable for a broader task-type statement.
- Codec-independent minimum description length or Kolmogorov complexity.
- A universal mechanism dictionary or compiler grammar.
- General FLOP, wall-clock, energy, RAM, VRAM, GPU/NPU, browser, or edge benefits.
- A confirmed compositional advantage in quantization-repair sample complexity.
- A universally Pareto-optimal autonomous consolidation controller.

## Claim discipline

Safe baseline statement:

> For some trained networks and task distributions, a composed input-output span admits a smaller task-preserving replacement than component-wise treatment under an explicit replacement grammar and decision rule. The effect was observed in the original residual-CNN program and directly tested under fresh locked protocols in a Small Vision Transformer and a residual MLP.

Do **not** translate this into “function composition always reduces mathematical complexity.”

Before broad announcement, rerun the integrated claim review after the final evidence-inclusion decisions described in `ANNOUNCEMENT_READINESS.md`.