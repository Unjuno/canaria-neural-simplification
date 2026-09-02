# Gaussian-shift interface dimension — C59–C61 imported handoff

Status: **research-only; not part of reviewed public `main` claim registry**.

## Provenance warning

This branch was created after the C59/C60 outcomes and after the C61 protocol had already been locked/executed externally. Therefore the GitHub timestamps in this branch are **not preregistration timestamps** for C59–C61.

The values below are an exact transcription of the project handoff supplied on 2026-09-02. Until the original runner, raw per-seed outputs, protocol files, and independent-audit artifacts are imported and hash-checked, this branch must distinguish:

- `IMPORTED_HANDOFF_RESULT`: outcome reported by the prior execution session, not independently reconstructed from GitHub artifacts here;
- `IMPORTED_LOCKED_PROTOCOL`: protocol reported as prospectively locked before fresh outcomes, but imported to GitHub later;
- `GITHUB_RECONSTRUCTED`: evidence independently recomputed from artifacts present in this repository branch.

No item is `GITHUB_RECONSTRUCTED` yet.

## Scientific question

Under the fixed Gaussian input shift `sigma = 0.04`, how small can the teacher-correction interface be while preserving the tested recursive replacement fidelity/utility criteria?

This line is distinct from C20/C21 SmallViT self-anchor experiments. In particular, C21 already confirmed an `8/32` self-anchored teacher correction on the SmallViT central two-block regime under a different protocol. Therefore this line must never be summarized as a universal statement that “SmallViT needs 16/32.”

Safe cross-architecture wording is restricted to the **Gaussian sigma=.04 protocol family**.

## C59 — Residual CNN P8 versus P16 confirmation

Evidence class: `IMPORTED_HANDOFF_RESULT`.

Decision reported by the prior session: `P8_NONINFERIOR_PASS`.

- fresh seeds: `47400–47415`;
- eligible: `16/16`;
- P8 joint success: `1.000`;
- P8−P16 validation accuracy: `-0.466 pp`, bootstrap95 `[-1.071,+0.055] pp`;
- preregistered validation non-inferiority margin: `-2 pp` -> PASS;
- P8/P16 NMSE geometric-mean ratio: `1.0387`, bootstrap95 `[1.0338,1.0435]`;
- preregistered NMSE-ratio margin: `1.25` -> PASS;
- P8 delta NMSE versus frozen: `-0.02288`, bootstrap95 `[-0.02467,-0.02098]`;
- teacher-shift safeguard: PASS;
- test data: not used;
- learned-parameter match: PASS;
- independent audit: reported PASS.

Scoped interpretation:

> In the imported Residual-CNN Gaussian-sigma=.04 cohort, P8/32 satisfied the prospectively defined non-inferiority gates relative to P16/32.

This is not an equality claim and not a universal minimum-interface claim.

## C60 — P4 versus P8 exploration

Evidence class: `IMPORTED_HANDOFF_RESULT`; exploratory only.

Reported decision: `ADVANCE_P4_TO_C61`.

- eligible: `16/16`;
- P4 joint success: `1.000`;
- P4−P8 validation accuracy: `-0.182 pp`, bootstrap95 `[-0.506,+0.145] pp`;
- P4/P8 NMSE geometric-mean ratio: `1.02537`, bootstrap95 `[1.02225,1.02877]`;
- P4 delta NMSE versus frozen: `-0.01100`, bootstrap95 `[-0.01267,-0.00950]`;
- teacher shift: PASS;
- all advance gates: PASS;
- independent audit: reported PASS.

C60 selected P4 for fresh confirmation. It does **not** confirm that P4 is sufficient.

## C61 — fresh P4 confirmation

Evidence class: `IMPORTED_LOCKED_PROTOCOL`; fresh outcome currently **unknown in this repository**.

Reported locked conditions:

- fresh seeds: `49400–49415`;
- comparison: P4 versus P8;
- calibration samples: `192`, fixed;
- Gaussian shift: `sigma=.04`, fixed;
- same nested QR basis;
- same calibration subset across dimensions;
- validation non-inferiority margin: `-2 pp`;
- NMSE ratio margin: `1.25`;
- minimum eligible models: `8`;
- paired bootstrap: `100000` resamples;
- held-out test: not used.

Reported pre-fresh equivalence check on seed `48400`: teacher, frozen, full32, all eight P4 replicates, all eight P8 replicates, and subset hashes matched exactly with zero numerical difference.

This equivalence check is implementation verification only; it is not fresh scientific confirmation.

## Current architecture-specific hypothesis

Current evidence motivates, but does not establish, the hypothesis that the minimum teacher-correction dimension under distribution shift depends on architecture and/or boundary representation geometry rather than being a universal fixed fraction of hidden width.

Potential confounds that must be separated before a general architecture claim:

- boundary location;
- candidate/replacement grammar;
- activation geometry and effective rank;
- teacher Jacobian spectrum;
- shift direction relative to the correction subspace;
- calibration-sample geometry;
- optimization conditioning.

## Promotion gate

Do not merge this line into public claims until all of the following are true:

1. original C59/C60/C61 runner(s) are imported;
2. original protocol/audit/result artifacts are imported with hashes;
3. C59/C60 aggregate values are independently reconstructed from raw seed rows;
4. C61 fresh cohort is complete and evaluated only by the locked rules;
5. a separate review explicitly reconciles this line with C20/C21 and other interface-dimension evidence;
6. public wording remains protocol-scoped and does not infer a universal architecture ratio from P4/P8/P16 labels.
