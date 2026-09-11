# Announcement candidate — 2026-09-12

**Status: REVIEWED PUBLICATION CANDIDATE — NOT MERGED, NOT RELEASED, NOT ANNOUNCED**

This document is a candidate public surface assembled after the post-v0.2 independent claim review. It does not replace the historical archive and does not close Issue #87.

## Candidate headline evidence

### Fresh direct digits baseline: R87R2

Under the fixed residual-MLP / sklearn-digits / first-two-block replacement protocol and the tested `portable_v2_sqrt64` CPU numerical profile, the prospectively fixed 16-model-seed cohort selected a smaller composed replacement budget on average while the selected test-accuracy difference satisfied the prespecified 2 percentage-point noninferiority margin.

Persisted decision:

- 16/16 eligible fresh model seeds (`871200`–`871215`)
- geometric composed/component-wise selected-budget ratio: `0.5316098878`
- mean log2 budget-ratio 95% paired bootstrap CI: `[-1.0412593748, -0.75]`
- composed selected a strictly smaller budget in 15/16 seeds; one tie
- selected test-accuracy difference (composed − component-wise) 95% CI: `[+0.0002777874, +0.0075000116]`
- decision: `R87R2_CONFIRMATORY_PASS`

Boundary: these are 16 model seeds on one reused digits dataset split, not 16 independent datasets. The test split did not select replacement budgets, but it is an existing reused test split. The numerical recipe is a scoped tested CPU research profile, not a universal backend portability result.

Most importantly, **R87R2 is not a recovery of the historical seeds 1200–1207 archive**. Exact reproduction of that archived cohort remains unresolved under Issue #87. The historical record must not be silently overwritten with R87R2 values.

Evidence:

- `results/reproduction/r87r2_fresh_sqrt_profile/PROTOCOL.json`
- `results/reproduction/r87r2_fresh_sqrt_profile/FRESH_ROWS.json`
- `results/reproduction/r87r2_fresh_sqrt_profile/DECISION.json`

## Bounded regression external-validity support: California Housing Phase4

On one fixed California Housing split, using the same residual-MLP family and a teacher recipe selected prospectively in Stage A without evaluating the held-out test split, the fresh eight-model-seed Stage-B cohort selected smaller composed replacement budgets in 8/8 seeds and satisfied the prespecified selected held-out-test R² noninferiority gate.

Persisted decision:

- 8/8 eligible fresh model seeds (`2900`–`2907`)
- teacher held-out-test R² mean 95% CI: `[0.7910913714, 0.7995540823]`
- geometric composed/component-wise selected-budget ratio: `0.4760285231`
- mean log2 budget-ratio 95% paired bootstrap CI: `[-1.3282417852, -0.8325187496]`
- composed selected a strictly smaller budget in 8/8 seeds
- selected held-out-test R² difference (composed − component-wise) 95% CI: `[+0.0043588589, +0.0096701812]`
- decision: `PHASE4_CONFIRMATORY_PASS`
- second-host scientific records: exact technical replication PASS

Boundary: this is one dataset and one fixed split, not eight independent datasets. The architecture remains residual-MLP. The result supports limited task/dataset external validity; it does not establish architecture universality. Selected budgets are minima only over the locked finite grid.

Evidence:

- `results/phase4/california_regression/STAGE_A_PROTOCOL.json`
- `results/phase4/california_regression/STAGE_A_RESULT.json`
- `results/phase4/california_regression/STAGE_B_CONFIRMATORY_PROTOCOL.json`
- `results/phase4/california_regression/stage_b_confirmatory/FRESH_ROWS.json`
- `results/phase4/california_regression/stage_b_confirmatory/DECISION.json`
- `results/phase4/california_regression/stage_b_confirmatory/TECHNICAL_REPLICATION.json`

## Boundary and negative evidence retained

The independent review excludes Phase3B from a positive stronger-teacher headline because its preregistered teacher-strength gates were uncertain. Phase3C remains negative/boundary evidence: within its locked same-architecture recipe grid, training-only nested selection found no teacher satisfying the preregistered stability/quality gate, and the outer test was not evaluated. Phase2E remains invalidated because of its implementation bug and must not be used for inference.

## Claims this candidate does not make

This candidate does **not** claim independent external reproduction, external peer review, a universal composition law, whole-model compression, LLM applicability, universal minimum interface dimension, runtime speedup, RAM/VRAM reduction, energy reduction, or hardware-wide portability. Existing systems measurements remain bounded prototype evidence only.

## Publication gate

Before any merge, release, or announcement:

1. `tools/audit_publication.py` must continue to pass its historical-preservation mode; its strict Issue-#87 reproduction path is not weakened or redefined.
2. `tools/audit_publication_candidate.py` must PASS on the exact candidate head, including Git-blob identity checks and raw-row recomputation for R87R2 and Phase4.
3. Repository CI must pass on the exact candidate head.
4. README / README.ja / STATUS / claims-and-evidence / announcement-readiness surfaces must be edited consistently with the reviewed ledger and re-audited.
5. Merge, release tagging, announcement, and Issue #13 closure remain separate explicit decisions.

Machine-readable review and candidate policy:

- `publication/POST_V02_CLAIM_LEDGER.json`
- `publication/CANDIDATE_CLAIM_POLICY.json`
- `publication/CANDIDATE_EVIDENCE_MANIFEST.json`
