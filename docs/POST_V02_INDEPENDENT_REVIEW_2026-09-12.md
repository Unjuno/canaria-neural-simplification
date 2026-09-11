# Post-v0.2 independent scientific claim review — 2026-09-12

This review answers Issues #96/#97. It is an internal independent re-review with fresh recomputation from persisted raw rows and git chronology. It is **not external peer review** and does not itself merge any research branch, create a release, or announce the project.

Review base: `main@c7fd4fb701064c9aeef5cc8109231d884e5258aa`.

Machine-readable ledger: `publication/POST_V02_CLAIM_LEDGER.json`.

Independent executable audit: `tools/review_post_v02_claims.py`; first successful workflow run: `34634973462`.

## Review method

For R87R2, Phase3B and Phase4, the reviewer script reads persisted `FRESH_ROWS.json` directly from the reviewed research refs, reconstructs selected-budget log ratios and selected test differences, independently regenerates the locked paired percentile bootstrap, and compares the recomputed intervals to the persisted decisions. It does not import the experiment evaluators.

For R87R2, Phase3B, Phase3C and Phase4, git history is checked so that protocol locks precede outcome files. Phase3C additionally requires the pre-outcome correction amendment to precede the result. Seed sets, missing/duplicate endpoints and test-selection flags are checked where applicable.

The first full audit passed. Verified chronology includes:

- R87R2 protocol add `091ebc2bd57681de9e8f14fe8f87ebeac16890c6` before persisted fresh outcome `88b557958b7274102cb6767d0e9eb61ddf1f261d`.
- Phase3B protocol `9c834076a40ec988f7ace799150515061c7ec6c8` before outcome `38acc726047791e81154544a9e6f6a1aa962136c`.
- Phase3C protocol `acca81d52f54e0c0ec7ca150bd4dc09784359159`, then pre-outcome amendment `59e478158a8b2999cb41dc69bceac3ced997f406`, then result `54cd79fc7b254c5d99576ed5c5b448580cc45138`.
- Phase4 Stage A protocol `a1c9ff1691b4b84f45c1b812e1fd6b6ecefeea6b` before result `42792c8d10e405de3919cf22a87afae30e16eef0`; Stage B protocol `4dade6ea65db5dc5dcb11d5f3e6f38f2e2ee6a0b` before fresh outcome `642e89468369c663afbeec02be19cf7716e0db42`.

## Decision ledger

| Evidence family | Review decision | Public role |
|---|---|---|
| R87R2 fresh sqrt-profile confirmation | **KEEP** | Primary reproducible direct-baseline candidate |
| Historical digits 1200–1207 archive | **EDIT** | Historical recorded result; not the current exact-reproduction baseline |
| Phase4 California Housing | **KEEP** | Bounded regression/dataset external-validity support |
| Phase3B stronger-teacher diabetes | **EXCLUDE from positive headline** | Keep as boundary evidence; overall decision remains UNCERTAIN |
| Phase3C nested-CV diabetes teacher search | **KEEP** | Negative/boundary evidence |
| SmallViT direct comparison already on main | **KEEP** | Supporting architecture-family evidence with existing test-isolation caveat |
| C61R–C77E and QR diagnostics | **EXCLUDE from headline** | Research/mechanism appendix only |
| Imported C59/C60 + original C61 | **EXCLUDE** | Imported provenance; original C61 unresolved |
| Systems S1–S7 | **EDIT** | CPU/runtime/serialization PoC only |
| Phase2E | **INVALIDATE** | Historical `DO_NOT_USE_FOR_INFERENCE` |
| Phase2I causal attribution | **INVALIDATE** | Retraction remains |
| Phase2O repair-sample advantage | **EXCLUDE** | Advantage not established |

## R87R2 — KEEP as the current reproducible direct baseline

### H

Under the fixed residual-MLP/digits/first-two-block replacement protocol and separately specified `portable_v2_sqrt64` CPU numerical profile, composed-span replacement should select a smaller passing learned-parameter budget on average while selected test accuracy remains noninferior within 2 percentage points.

### T / D

Fresh seeds `871200–871215`, all retained. Validation selects the first passing endpoint; test does not select budgets. Locked paired bootstrap: 100,000 resamples, seed `87122026`.

Independent recomputation reproduces:

- `R87R2_CONFIRMATORY_PASS`;
- geometric composed/component-wise selected-budget ratio `0.531609887796847`;
- log2 ratio 95% CI `[-1.041259374819711, -0.75]`;
- selected composed-minus-component-wise test-accuracy 95% CI `[0.000277787446975708, 0.007500011567026376]` in fraction units;
- composed lower in 15/16 and tied in 1/16.

### C / U

This result uses one reused digits split and model seeds, not 16 datasets. The numerical profile is a scoped research implementation, not a universal bitwise CPU/GPU theorem. Most importantly, it is **not historical archive recovery**.

**Review decision:** promote this only as a *new reproducible baseline*. Do not claim that the historical 1200–1207 numerical archive was reproduced.

## Historical core digits archive — EDIT, not erase

The old archived result remains scientifically and historically relevant: the direction has repeatedly persisted, but exact archived endpoints/statistics were not reproduced in current environments. Issue #87 therefore remains open.

The public surface should stop using the historical numbers as the only reproducible headline. Instead:

1. R87R2 becomes the current reproducible direct evidence.
2. Historical 1200–1207 values remain explicitly labeled recorded historical values.
3. The unresolved old producing environment/provenance stays visible under Issue #87.

This changes **claim selection**, not historical data.

## Phase4 California Housing — KEEP as bounded external validity

### H

A larger real regression dataset with a competent teacher should retain the direct composition-budget advantage if it is not merely an artifact of weak-teacher diabetes regression.

### T / D

Stage A selected `low_lr_30` without test use. Stage B used fresh seeds `2900–2907`, all retained, and first validation-passing endpoints. Independent recomputation reproduces:

- `PHASE4_CONFIRMATORY_PASS`;
- teacher held-out test R2 95% CI `[0.7910913713697723, 0.7995540822785517]`, above the locked 0.70 gate;
- composed/component-wise geometric selected-budget ratio `0.47602852309683874`;
- log2 ratio 95% CI `[-1.328241785222183, -0.832518749639422]`;
- composed lower in 8/8;
- selected test R2 difference 95% CI `[0.004358858924945988, 0.00967018123690487]`, passing the locked -0.03 noninferiority margin.

### C / U

This is one fixed California Housing split and the same residual-MLP family. It supports task/dataset external validity, **not architecture universality**. It does not imply runtime, RAM, VRAM, energy, LLM-scale, or universal complexity gains.

**Review decision:** eligible as a supporting public claim after the evidence is selectively integrated into the publication candidate.

## Phase3B / Phase3C — retain the failure mode, do not hide it

Phase3B independently recomputes to `PHASE3B_CONFIRMATORY_UNCERTAIN`: the composition-budget and selected-utility gates pass, but both teacher-strength gates are uncertain. It therefore cannot be advertised as a successful stronger-teacher regression confirmation.

Phase3C then used training-only multi-fold teacher selection and found no eligible same-architecture recipe within the locked grid. The outer test was not evaluated. This is useful negative evidence: it shows why the diabetes line was not rescued by post-hoc threshold weakening or recipe expansion.

## Other post-v0.2 evidence

The existing main decision to keep C61R–C77E, QR diagnostics and recursive composition out of the headline remains correct. These studies inform mechanism and boundaries but do not support universal minimum-interface, whole-model reduction, communication, or cross-architecture theorems.

The imported Residual-CNN C59/C60 line remains separate from the Residual-MLP R-line; original C61 remains unresolved.

Systems S1–S7 remain PoC evidence only. No public RAM/GPU/VRAM/energy claim should be inferred from serialization size or one CPU workload.

Phase2E remains invalidated by implementation bug and must never be used as inferential evidence. Phase2I attribution remains retracted; Phase2O did not establish the claimed advantage.

## Announcement-readiness consequence

Issue #87 should remain **open** as historical provenance/exact-reproduction debt. However, after this review it does **not need to block a new scoped headline whose primary numerical evidence is R87R2**, provided public docs clearly distinguish the new baseline from the unrecovered archive.

Issue #13 should **not close yet**. Remaining work is editorial/integration and executable gating rather than more same-family experiments:

1. selectively integrate reviewed R87R2 and Phase4 evidence/provenance into a publication-candidate branch;
2. update README, STATUS, `docs/CLAIMS_AND_EVIDENCE.md`, `docs/ANNOUNCEMENT_READINESS.md`, and `publication/CLAIM_POLICY.json` consistently;
3. modify the executable publication gate so that it validates the selected new baseline rather than demanding historical archive parity as the only possible PASS;
4. run repository audit, publication audit and final-head CI;
5. keep external peer review and independent external reproduction explicitly unclaimed.

## Final review outcome

**PASS_WITH_PUBLIC_SURFACE_EDITS_REQUIRED.**

The evidence base is sufficient to stop adding same-family experiments for announcement readiness. The next step is a selectively assembled publication candidate, not automatic merging of Draft PRs #90/#91/#93/#95.
