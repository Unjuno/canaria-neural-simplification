# Claims and evidence: reviewed research-preview scope

This registry reflects the post-v0.2 independent claim review plus the PR-level publication disposition required by Issue #13. It changes the **editorial evidence selection**, not any locked scientific outcome. The historical 1200–1207 archive remains preserved and its exact modern reproduction remains unresolved; see [REPRODUCTION_DISCREPANCY.md](REPRODUCTION_DISCREPANCY.md).

The machine-readable disposition ledger is [POST_V02_CLAIM_LEDGER.json](../publication/POST_V02_CLAIM_LEDGER.json). The open-PR publication map is [OPEN_PR_DISPOSITION_2026-09-12.json](../publication/OPEN_PR_DISPOSITION_2026-09-12.json). The active publication policy is [CLAIM_POLICY.json](../publication/CLAIM_POLICY.json).

## Primary current direct baseline: R87R2 — KEEP

> Under the fixed residual-MLP / sklearn-digits / first-two-block replacement protocol and the tested `portable_v2_sqrt64` CPU numerical profile, a prospectively fixed fresh 16-model-seed cohort selected a smaller composed replacement budget on average while selected test accuracy satisfied the prespecified 2 percentage-point noninferiority margin.

Evidence: [protocol](../results/reproduction/r87r2_fresh_sqrt_profile/PROTOCOL.json), [fresh raw rows](../results/reproduction/r87r2_fresh_sqrt_profile/FRESH_ROWS.json), [persisted decision](../results/reproduction/r87r2_fresh_sqrt_profile/DECISION.json), [vendored-evidence manifest](../publication/CANDIDATE_EVIDENCE_MANIFEST.json), [executable candidate audit](../tools/audit_publication_candidate.py).

Reviewed metrics:

- 16/16 eligible fresh model seeds (`871200`–`871215`)
- geometric composed/component-wise selected-budget ratio `0.5316098878`
- mean log2 budget-ratio 95% paired bootstrap CI `[-1.0412593748, -0.75]`
- composed strictly lower in 15/16 seeds; one tie
- selected test-accuracy difference, composed minus component-wise, 95% CI `[+0.0002777874, +0.0075000116]`
- locked decision `R87R2_CONFIRMATORY_PASS`

Required boundary: these are 16 model seeds on one reused digits split, not 16 datasets. The selected budgets are finite-grid minima, not mathematical minima. Test metrics do not select replacement budgets, but the test split is reused. The scoped CPU numerical profile is not a universal portability theorem. No runtime, RAM, VRAM, energy, whole-model, or LLM claim follows.

## Historical digits 1200–1207 — EDIT / retain as historical record

The historical recorded cohort showed the same directional composition-budget pattern: mean selected replacement parameters 3,584 component-wise versus 1,728 composed, with composed lower in 8/8 recorded seeds. Those archived values remain part of the historical record.

However, current full-cohort reruns did not exactly reproduce all archived per-seed endpoints/statistics. Therefore:

- do not call the historical archive exactly reproduced;
- do not overwrite historical values with R87R2 values;
- do not use R87R2 to close Issue #87;
- preserve [the disclosed discrepancy](REPRODUCTION_DISCREPANCY.md).

R87R2 is a new confirmatory cohort under a separately scoped numerical profile. The independent review permits it to replace the historical cohort as the **current direct headline evidence**, not to retroactively repair the archive.

## Bounded regression external-validity support: California Housing Phase4 — KEEP

> On one fixed California Housing split, using the same residual-MLP family and a teacher recipe selected prospectively in Stage A without held-out-test evaluation, a fresh eight-model-seed confirmatory cohort selected smaller composed replacement budgets in 8/8 seeds and satisfied the prespecified selected held-out-test R² noninferiority gate.

Evidence: [Stage-A protocol](../results/phase4/california_regression/STAGE_A_PROTOCOL.json), [Stage-A result](../results/phase4/california_regression/STAGE_A_RESULT.json), [Stage-B protocol](../results/phase4/california_regression/STAGE_B_CONFIRMATORY_PROTOCOL.json), [fresh raw rows](../results/phase4/california_regression/stage_b_confirmatory/FRESH_ROWS.json), [decision](../results/phase4/california_regression/stage_b_confirmatory/DECISION.json), [technical replication](../results/phase4/california_regression/stage_b_confirmatory/TECHNICAL_REPLICATION.json).

Reviewed metrics:

- 8/8 eligible fresh model seeds (`2900`–`2907`)
- teacher held-out-test R² mean 95% CI `[0.7910913714, 0.7995540823]`
- geometric composed/component-wise selected-budget ratio `0.4760285231`
- mean log2 budget-ratio 95% paired bootstrap CI `[-1.3282417852, -0.8325187496]`
- composed strictly lower in 8/8 seeds
- selected held-out-test R² difference, composed minus component-wise, 95% CI `[+0.0043588589, +0.0096701812]`
- second-host exact scientific-record replication PASS
- locked decision `PHASE4_CONFIRMATORY_PASS`

Required boundary: this is one dataset and one fixed split, not eight independent datasets. The architecture remains residual-MLP. It supports bounded task/dataset external validity, not architecture universality. Selected budgets are minima only over the locked finite grid.

## Older diabetes regression PR #11 — valid bounded result, excluded from current headline

PR #11's locked primary composition-budget result remains a valid research result under its exact protocol. Recalculation from the persisted eight confirmatory seed rows reproduces mean `log2(B_composed/B_componentwise) = -1.0263620978`, paired-bootstrap 95% CI `[-1.1842413985, -0.8684827971]`, and geometric budget ratio `0.4909465609`.

It is not selected as the current regression-support headline because teacher held-out-test R² ranges only from about `0.112` to `0.255`, no teacher-eligibility filter was preregistered, and component-wise endpoints for seeds 2201 and 2204 reach the locked `8192`-parameter grid ceiling. This does not invalidate PR #11; it limits its interpretability. Phase4 is the selected current bounded regression support because it prospectively establishes a competent teacher before the fresh replacement cohort.

## Supporting architecture-family evidence: SmallViT — KEEP with narrow scope

[SmallViT direct composition](CROSS_FAMILY_COMPOSITION_REPLICATION.md) remains supporting evidence for a declared central two-block span on digits. It is not full-Transformer or LLM evidence. Its runner recorded test metrics for every candidate, so its operational test-isolation boundary differs from R87R2 and Phase4.

## Negative and boundary evidence

| Evidence | Review disposition | Public interpretation |
|---|---|---|
| Phase3B stronger-teacher confirmatory | **EXCLUDE from positive headline** | Composition-budget and selected-utility gates passed, but both preregistered teacher-strength gates were uncertain. Do not advertise a positive stronger-teacher external-validity result. |
| Phase3C nested-CV teacher selection | **KEEP as negative/boundary evidence** | No teacher in the locked 13-recipe same-architecture grid satisfied the preregistered stability/quality gate; the outer test was not evaluated. This does not prove that a strong diabetes teacher is impossible. |
| Phase2E | **INVALIDATE** | `INVALIDATED_IMPLEMENTATION_BUG` / `DO_NOT_USE_FOR_INFERENCE`; preserve history, never use for positive or negative inference. |
| Phase2I causal attribution | **INVALIDATE** | Attribution remains retracted. |
| Phase2O repair-sample advantage | **EXCLUDE as established advantage** | Advantage was not established. |

## Research-only / provenance-only families

| Family | Public role |
|---|---|
| Imported C59/C60 Residual-CNN | Imported handoff evidence only; architecture boundary must be explicit. |
| Original C61 | Outcome unresolved; do not infer it from C61R. |
| C61R–C77E and QR diagnostics | Research appendix/mechanism/boundary evidence; no universal minimum-interface or communication theorem. |
| Recursive-composition extensions | Research only; no blanket cross-architecture theorem. |
| Systems S1–S7 | CPU/serialization/runtime prototype evidence only; no general RAM, GPU/VRAM, energy, or physical-device claim. |

## Open-PR publication policy

All 30 open PRs in the 2026-09-12 review snapshot have an explicit publication disposition. This does not authorize closing or merging them. Research PRs are not mechanically merged into the publication candidate; reviewed evidence is selectively vendored or referenced according to the disposition ledger. CI performs a live GitHub check and fails if a currently open PR has no disposition. Any new scientific PR opened after the snapshot requires a separate inclusion/exclusion decision before announcement.

## Interpretation rules

Noninferiority is not equality or superiority. A model-seed bootstrap is conditional on the tested dataset/split and does not quantify dataset-level uncertainty. Replacement parameters, correction dimension, serialized bytes, FLOPs, latency, RAM, VRAM, and energy are distinct quantities.

No universal complexity law, universal minimum dimension, broad architecture transfer, whole-model compression, general LLM applicability, or general hardware benefit is claimed.

## Publication verification

Run:

```bash
python tools/audit_publication.py
python tools/audit_publication_candidate.py
python tools/audit_open_pr_disposition.py
```

CI additionally runs `tools/audit_open_pr_disposition.py --live-github`. The candidate audit verifies exact reviewed Git-blob identities and recomputes R87R2 and Phase4 decisions from persisted raw rows. The separate historical Issue-#87 gate remains available as `python tools/audit_publication.py --require-reproduction`; it is intentionally not redefined by the fresh R87R2 result.
