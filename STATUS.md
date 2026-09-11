# Project status

**Mode: reviewed publication candidate; evidence gate passed on the pre-surface candidate, but this branch is not merged, released, or announced.**

The post-v0.2 independent claim review selected `R87R2_FRESH_SQRT_PROFILE_CONFIRMATION` as the current reproducible direct baseline candidate and California Housing Phase4 as bounded regression external-validity support. The candidate branch vendors the reviewed evidence byte-identically from the source research commits and recalculates the locked decisions from persisted raw rows.

The repository is already public. Publication-candidate status is not the same as merge, release, announcement, independent external reproduction, or peer review. Issue #13 remains the readiness tracker.

## Current direct baseline candidate: R87R2

R87R2 is a prospective 16-model-seed residual-MLP / digits / first-two-block confirmation under the scoped `portable_v2_sqrt64` CPU numerical profile.

- 16/16 eligible fresh seeds, `871200`–`871215`
- geometric composed/component-wise selected-budget ratio: `0.5316098878`
- mean log2 budget-ratio 95% paired bootstrap CI: `[-1.0412593748, -0.75]`
- composed selected a strictly smaller budget in 15/16 seeds; one tie
- selected test-accuracy difference 95% CI, composed minus component-wise: `[+0.0002777874, +0.0075000116]`
- locked decision: `R87R2_CONFIRMATORY_PASS`

This is one reused digits split with 16 model seeds, not 16 datasets. It does not establish a universal minimum, universal backend portability, runtime benefit, or whole-model compression.

Evidence: [protocol](results/reproduction/r87r2_fresh_sqrt_profile/PROTOCOL.json), [raw rows](results/reproduction/r87r2_fresh_sqrt_profile/FRESH_ROWS.json), [decision](results/reproduction/r87r2_fresh_sqrt_profile/DECISION.json).

## Historical core archive remains unresolved separately

The historical seeds 1200–1207 remain preserved. Modern full-cohort reruns retained the directional pattern but did not exactly reproduce all archived endpoints/statistics. [REPRODUCTION_DISCREPANCY.md](docs/REPRODUCTION_DISCREPANCY.md) records the discrepancy.

R87R2 is not historical recovery. Issue #87 remains open as historical provenance/reproduction debt. The strict historical gate is still available with `python tools/audit_publication.py --require-reproduction` and remains blocked while Issue #87 is unresolved. The independent review explicitly determined that this historical debt does not, by itself, block using the separately preregistered fresh R87R2 cohort as the current scoped headline evidence.

## Bounded external-validity support: Phase4

California Housing Phase4 used one fixed dataset split and the same residual-MLP family. Stage A selected a competent teacher recipe without held-out-test evaluation; Stage B then ran eight fresh model seeds under a locked confirmatory protocol.

- 8/8 eligible
- geometric composed/component-wise selected-budget ratio: `0.4760285231`
- mean log2 budget-ratio 95% CI: `[-1.3282417852, -0.8325187496]`
- selected held-out-test R² difference 95% CI: `[+0.0043588589, +0.0096701812]`
- teacher held-out-test R² mean 95% CI: `[0.7910913714, 0.7995540823]`
- exact second-host technical replication
- locked decision: `PHASE4_CONFIRMATORY_PASS`

This supports limited task/dataset external validity, not architecture universality. [Phase4 decision](results/phase4/california_regression/stage_b_confirmatory/DECISION.json)

## Independent review dispositions

The machine-readable [post-v0.2 claim ledger](publication/POST_V02_CLAIM_LEDGER.json) records the review disposition. Important boundaries include:

- R87R2: **KEEP** as current direct baseline candidate, not archive recovery.
- Phase4: **KEEP** as bounded regression external-validity support.
- Phase3B: **EXCLUDE** from a positive stronger-teacher headline; teacher-strength gates were uncertain.
- Phase3C: **KEEP** as negative/boundary evidence; no qualifying teacher was selected under its locked grid and the outer test was not evaluated.
- Phase2E: **INVALIDATE** for inference because of the implementation bug.
- Imported C59/C60 remain Residual-CNN provenance; original C61 remains unresolved and is not repaired by the Residual-MLP C61R line.
- Systems S1–S7 remain bounded prototype measurements, not general RAM/GPU/VRAM/energy claims.

## Executable candidate gate

The active publication integrity gate is:

```bash
python tools/audit_publication.py
```

The scientific candidate evidence gate is:

```bash
python tools/audit_publication_candidate.py
```

It checks reviewed Git-blob identities and recomputes R87R2 and Phase4 aggregate decisions from the vendored raw rows. The first candidate evidence-gate run succeeded before these public-surface edits; the exact final candidate head must pass again before any merge decision.

## Preservation

- Frozen tag: `v0.2.0-public-snapshot`, commit `556dce21c7a5516a16780cb28d528d1ff3968e53`; never rewritten.
- Pre-publication-preparation baseline: `41872aca00ee5750556c93e114f705bce2e9c611`.
- Review base main: `c7fd4fb701064c9aeef5cc8109231d884e5258aa`.
- Historical migrations remain hash-checked by `publication/PRESERVATION_MANIFEST.json`.
- No historical outcome is rewritten by selecting a new current headline cohort.

## Remaining publication work

The exact final candidate head must pass repository and publication CI; the full README/STATUS/claims/readiness surface must be reviewed for internal consistency; the Draft publication PR must receive independent review. Merge, release tag, announcement, and Issue #13 closure remain separate explicit actions. None has occurred merely because the evidence gate passed.

Open research directions include independent-dataset replication, broader architecture families, physically constrained teacher-feedback interfaces, and fixed-device resource measurements. They are not prerequisites for this narrowly scoped preview because those stronger capabilities are not claimed.
