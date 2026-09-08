# Claims and evidence: research-preview scope

This is the current editorial selection for the research preview. It **narrows the headline**, does not change locked outcomes, and does not promote unmerged research because it reports PASS. The previous registry is preserved [byte-for-byte](../archives/reviews/CLAIMS_AND_EVIDENCE_2026-08-26.md); its decision history is [here](INDEPENDENT_REREVIEW_2026-08-26.md).

> **New qualification (2026-09-08):** the complete baseline rerun did not strictly match all historical numerical values. [Discrepancy](REPRODUCTION_DISCREPANCY.md). The numbers below are historical recorded results, not a guarantee of present exact reproduction.

## Proposed headline: bounded baseline with disclosed reproduction status

> In the specified residual-MLP / digits / first-two-block experiment, fitting one composed input–output replacement selected a smaller passing learned-parameter budget than fitting replacements component-wise, under the declared grid and validation rule.

Evidence: [protocol](../results/core_discovery_digits/PROTOCOL_LOCK.json), [complete recorded summary](../results/core_discovery_digits/confirm_summary.json), [unchanged runner](../scripts/reproduce/core_discovery_digits/run_confirmatory.py), [interpretation](CORE_DISCOVERY_REPLICATION_DIGITS.md), [reproduction](../QUICKSTART.md).

The original eight model seeds are 1200–1207. Mean selected replacement budgets: 3584 component-wise and 1728 composed; composed lower in 8/8. The geometric paired budget ratio is 0.482339. “Minimum” is the smallest passing tested grid point, not a true function-complexity minimum. Validation chooses the endpoint; candidate test is evaluated at the selected endpoint; teacher and prespecified-control test metrics are also recorded. A shared fixed dataset split limits external validity. A repeat of these seeds is reproduction, not eight new confirmatory models.

## Supporting baseline, not additional publication reruns

| Evidence | Retained scope and caveat |
|---|---|
| [SmallViT direct comparison](CROSS_FAMILY_COMPOSITION_REPLICATION.md) | Digits and a declared two-block span; the locked selector excludes test, but the runner recorded test metrics for every candidate. This is weaker operational test isolation. |
| [Training-time studies](TRAINING_TIME_CONSOLIDATION.md) and [later controls](LATE_STAGE_FINDINGS.md) | Small character-LM schedules and update/parameter proxies; no broad LLM or measured energy claim. |
| [Phase 2 precision](phase2/README.md) | A–C have public runners/results; later raw-artifact availability is incomplete. Quantizer-specific evidence is not native hardware FP4/FP8 support. |
| [CPU runtime PoC](RUNTIME_POC.md) | One small recorded workload. Serialization observations are not total RAM, GPU/VRAM or deployment-device evidence. |

## Newer research: inspect, do not silently promote

The [research index](RESEARCH_INDEX.md) pins the original sources and describes the evidence classes. In particular, C76R reports relative non-inferiority of head-derived9 versus P64 at sigma0.36/N384; it is not a minimum-dimension predictor, a new linear-algebra theorem, a 64-to9 whole-model reduction, or a measured teacher-communication saving. Both final mappings retain 4096 parameters. Full teacher residuals are computed during calibration. C76R's candidate-to-teacher -5pp absolute criterion was not established in its cohort. C77E is a separate exploratory study.

## Explicit exclusion decisions for this preview

| Family | Decision |
|---|---|
| Original Residual-CNN program and v10–v25 history | Archive/supporting provenance; not the clean headline reproduction. |
| Recursive-composition C1–C25 and later extensions | Research index only; no blanket cross-architecture theorem. |
| Systems S1–S7 | Research PoC only; no physical-device resource headline. |
| Imported C59/C60 and original C61 | Imported results/protocol only; original C61 unresolved, not repaired by the differently architected C61R. |
| C61R–C77E and QR order diagnostics | Separately classed research; no retroactive preregistration, no automatic baseline promotion. |
| Phase 3/3B regression work | Not a headline external-validity result; review of target quality remains separate. |
| Phase 2E | `INVALIDATED_IMPLEMENTATION_BUG`; `DO_NOT_USE_FOR_INFERENCE`. Preserved, not valid negative evidence. |
| Phase 2I causal attribution / Phase 2O repair-sample advantage | Attribution retracted / advantage not established. |

## Interpretation rules

Non-inferiority is not equality. Failure to establish non-inferiority is not automatically established inferiority. Use the original protocol decision unchanged; explain uncertainty rather than relabeling old files. Model-seed confidence intervals do not quantify dataset-level uncertainty. Replacement parameters, correction rank, artifact bytes, timing, RAM and energy are different quantities.

No universal complexity law, codec-independent minimum, broad architecture transfer, or general hardware benefit is advertised. [Negative results](NEGATIVE_RESULTS.md) and [correction registry](../results/phase2/precision_composition/CORRECTION_STATUS.json) remain visible beside positive results.
