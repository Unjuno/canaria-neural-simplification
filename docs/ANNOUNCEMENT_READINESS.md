# Announcement readiness — scoped research preview

Issue #13 is the readiness tracker. Public repository visibility and the frozen `v0.2.0-public-snapshot` tag do not themselves pass this gate.

## Current outcome: REVIEWED CANDIDATE, NOT YET MERGED OR ANNOUNCED

The post-v0.2 independent claim review accepted `R87R2_FRESH_SQRT_PROFILE_CONFIRMATION` as the current reproducible direct baseline candidate and California Housing Phase4 as bounded regression external-validity support. The candidate branch vendors the reviewed evidence byte-identically, recalculates the locked decisions from persisted raw rows, and carries a clean-checkout R87R2 technical-reproduction workflow.

The older historical 1200–1207 cohort remains a separate unresolved reproduction problem. Modern reruns retained the directional pattern but not all archived endpoints/statistics; see [REPRODUCTION_DISCREPANCY.md](REPRODUCTION_DISCREPANCY.md). Issue #87 remains open. R87R2 is explicitly **not** a historical recovery.

The independent review determined that Issue #87 does not by itself block a narrowly scoped announcement that uses the separately preregistered fresh R87R2 cohort as the current direct baseline while disclosing the historical debt.

## Intended announcement scope

Release **research code and an auditable, bounded evidence collection**. The current headline may report the fresh R87R2 direct-composition result under its exact residual-MLP/digits/two-block/numerical-profile boundary. Phase4 may be reported as limited task/dataset external-validity support within the same residual-MLP family.

Do not announce a completed universal compression technology, universal minimum-dimension law, whole-model compression result, general LLM method, or general runtime/memory/energy improvement.

## Current publication gate

| Requirement | Required evidence |
|---|---|
| Independent claim selection | `publication/POST_V02_CLAIM_LEDGER.json` remains `PASS_WITH_PUBLIC_SURFACE_EDITS_REQUIRED`, with R87R2 KEEP, Phase4 KEEP, Phase3B excluded from positive headline, Phase3C retained as boundary evidence, and Phase2E invalidated. |
| Open-PR claim disposition | `publication/OPEN_PR_DISPOSITION_2026-09-12.json` gives every open PR in the 2026-09-12 snapshot an explicit publication role. CI queries GitHub and fails if a currently open PR is not classified. A new scientific PR opened after the snapshot requires explicit review before announcement. |
| R87R2 evidence integrity | Vendored protocol/raw rows/decision remain Git-blob identical to reviewed source evidence; fresh 16-seed cohort is complete; raw-row recalculation reproduces `R87R2_CONFIRMATORY_PASS`. |
| R87R2 clean-checkout technical reproduction | `.github/workflows/publication-headline-reproduction.yml` installs the locked Python/PyTorch/numerical dependency set, retrains the fixed seeds `871200`–`871215`, and requires exact equality of all 16 scientific outcome objects plus the aggregate decision with the reviewed vendored primary evidence. Repeating this cohort adds zero independent scientific seeds. |
| Phase4 evidence integrity | Stage-A/Stage-B evidence remains Git-blob identical; fresh eight-seed cohort is complete; raw-row recalculation reproduces `PHASE4_CONFIRMATORY_PASS`; exact technical replication remains PASS. |
| Historical honesty | Issue #87 stays open; historical 1200–1207 exact reproduction is not claimed; R87R2 is never labeled old-value recovery. |
| Public-surface consistency | README, README.ja, STATUS, QUICKSTART, CITATION, this file, claims registry, and `publication/CLAIM_POLICY.json` describe compatible evidence selection and boundaries. |
| Preservation | Historical/migrated evidence continues to match `publication/PRESERVATION_MANIFEST.json`; frozen tag remains unchanged. |
| Repository integrity | Repository audit, candidate publication audit, open-PR disposition audit, and headline clean-reproduction CI succeed on the exact final candidate head. |
| Review boundary | CI/audit is not described as independent external reproduction or peer review. |

Executable candidate checks without retraining:

```bash
python tools/audit_publication.py
python tools/audit_publication_candidate.py
python tools/audit_open_pr_disposition.py
```

CI additionally runs `audit_open_pr_disposition.py --live-github`, so an open PR absent from the disposition ledger blocks the candidate gate. The local form validates the fixed snapshot and its publication decisions.

For the actual clean-checkout headline retraining path, follow [QUICKSTART.md](../QUICKSTART.md) or the pinned `.github/workflows/publication-headline-reproduction.yml` workflow. The fixed environment is part of the scope; successful repetition is technical reproducibility of the already fixed cohort, not a larger scientific sample.

## Open scientific PR disposition

The 2026-09-12 snapshot contains 30 open PRs, and all 30 have an explicit publication disposition in `publication/OPEN_PR_DISPOSITION_2026-09-12.json`. Open status is therefore not treated as an implicit scientific endorsement or an unresolved headline decision.

Key decisions are:

- PR #90 / R87R2: selectively vendor the reviewed evidence as the primary current direct baseline; do not mechanically merge the research PR.
- PR #95 / Phase4: selectively vendor as bounded regression support; do not mechanically merge the research PR.
- PR #91 / Phase3B: exclude from a positive stronger-teacher headline; retain as boundary evidence.
- PR #93 / Phase3C: retain as negative/boundary evidence.
- PR #11 / original diabetes regression: keep as a valid bounded research result but exclude from the current headline. Its locked primary budget result independently recalculates, but teacher test R² is only about `0.112–0.255`, no teacher-eligibility filter was preregistered, and two component-wise endpoints reach the locked grid ceiling. Phase4 is the selected current regression support instead.
- PRs #69–#86: research appendix/mechanism/boundary evidence only; no universal minimum-interface or communication theorem.
- PR #68: imported C59/C60 provenance only, with original C61 unresolved.
- PR #56: selective supporting SmallViT / bounded systems evidence only; no wholesale integration merge for the announcement.
- PR #89: Issue-#87 numerical/provenance diagnostic only.
- PRs #14/#17: historical hardening/structure paths, not current claim-selection merge paths.
- PR #100 remains the independent internal claim-selection review prerequisite; PR #101 remains the publication candidate requiring independent review.

No PR is authorized for mechanical merge by this ledger, and the ledger does not close any PR.

## Historical strict reproduction gate remains separate

The pre-existing strict historical cohort path is retained:

```bash
python tools/audit_publication.py --require-reproduction
```

This path tests the historical seeds 1200–1207 reproduction report and remains blocked while Issue #87 is unresolved. It is a provenance/reproduction debt gate, **not** the scientific gate for the new R87R2 cohort. A future exact historical recovery would be useful, but current R87R2 evidence must not be relabeled as that recovery.

## Evidence currently admitted to the announcement surface

### Primary direct baseline: R87R2

- 16/16 eligible fresh model seeds
- geometric composed/component-wise selected-budget ratio `0.5316098878`
- mean log2 budget-ratio 95% CI `[-1.0412593748, -0.75]`
- composed strictly lower in 15/16 seeds, one tie
- selected test-accuracy difference 95% CI `[+0.0002777874, +0.0075000116]`
- one reused digits split; no independent-dataset claim
- no runtime/memory/energy inference

### Bounded support: California Housing Phase4

- 8/8 eligible fresh model seeds
- geometric composed/component-wise selected-budget ratio `0.4760285231`
- mean log2 budget-ratio 95% CI `[-1.3282417852, -0.8325187496]`
- selected held-out-test R² difference 95% CI `[+0.0043588589, +0.0096701812]`
- exact second-host technical replication
- one dataset / one split / same residual-MLP family; no architecture-universal claim

## Explicitly not promoted

The original PR #11 diabetes result is retained as valid bounded research evidence but is not the selected current regression support because of the weak/variable teacher regime. Phase3B is not a positive stronger-teacher headline because its teacher-strength gates were uncertain. Phase3C is negative/boundary evidence. Imported C59/C60 remain Residual-CNN provenance and original C61 remains unresolved. C61R–C77E and related QR diagnostics remain research appendix/mechanism evidence. Systems S1–S7 remain bounded prototypes. Phase2E remains `INVALIDATED_IMPLEMENTATION_BUG` / `DO_NOT_USE_FOR_INFERENCE`.

## Final actions still required

A PASS on the exact final candidate head is necessary but not sufficient. Before announcement, all exact-head technical gates—publication evidence audit, live open-PR disposition audit, repository audit, and clean-checkout R87R2 reproduction—must pass, and the publication-candidate PR must receive independent review after/with the independent evidence-selection PR. Merge, release tagging, social announcement, DOI/publication actions, and Issue #13 closure are separate decisions; none is performed automatically by CI or by this document.

Independent external reproduction and scientific peer review remain outstanding. They are not claimed or substituted by repository CI.

See [current status](../STATUS.md), [claim registry](CLAIMS_AND_EVIDENCE.md), [candidate announcement text](ANNOUNCEMENT_CANDIDATE_2026-09-12.md), and [active policy](../publication/CLAIM_POLICY.json).
