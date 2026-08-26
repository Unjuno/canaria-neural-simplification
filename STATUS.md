# Project status

**Current mode: publication-quality gate and repository-state cleanup. Broad experiment expansion is stopped.**

The frozen v0.2.0 **public snapshot** remains the version baseline. Post-snapshot precision work is staged separately on this research branch.

## Independent re-review

The review required by `REVIEW_HANDOFF.md` and Issue #9 was performed on 2026-08-26.

Review record:

- `docs/INDEPENDENT_REREVIEW_2026-08-26.md`
- `docs/CLAIMS_AND_EVIDENCE.md`

The review classified material claims as `KEEP`, `EDIT`, `REMOVE`, or `INVALIDATE` and changed the public surface accordingly.

Do not publish/post or mark PR #7 ready while Issue #9 remains open. Closing Issue #9 certifies this scientific quality gate only; the v0.2.0 tag/release boundary and PR #7 merge remain separate repository-state decisions.

## Core claim retained

The project-level thesis is **task-conditioned compositional simplification under explicit operational rules**:

> Some learned spans can admit smaller task-preserving replacements when fitted as one composed input-output function than when simplified at implementation-component boundaries.

The strongest fresh direct replication is the residual MLP with exact learned replacement-parameter matching:

- fresh seeds `1200–1207`;
- composed lower minimum passing budget in `8/8`;
- geometric composed/component-wise budget ratio `0.4823×`;
- test utility evaluated only after validation-selected endpoint choice.

The SmallViT direct replication is retained with a disclosed isolation caveat: its locked selection rule excludes test metrics, but its runner records test accuracy for all candidates. That is weaker operational test isolation than the residual-MLP runner.

## Training-time boundary

Retained:

- G7 primary: progressive consolidation beat preregistered early/late one-shot controls;
- G15/G17: intervening task learning, not merely factorized fitting, is part of the tested staged advantage;
- G19: staged-path effect also observed on `5→4→2` versus `5→2` with equal compiler-update counts;
- G18: the tested deadline-aware controller improved over the tested static controller;
- G20d/e and G22–G26: recontracting can make fitting easier while residual error becomes more task-sensitive in the small character-LM testbed.

G21 remains a valid failure and G27 remains exploratory/no-Pareto-claim.

## Phase 2 correction boundary

Supported:

- Phase 2A: 4-bit composed coded-size advantage under the locked residual-MLP quantizer/accounting;
- Phase 2B: increasing weight count alone did not rescue the tested naive 3-bit per-matrix PTQ;
- Phase 2C: row-wise scales rescued 3-bit PTQ for both topologies;
- corrected later work supports viability of short activation-domain QAT-style repair in the tested residual-MLP family.

Critical invalidation:

**Phase 2E is `INVALIDATED_IMPLEMENTATION_BUG` and `DO_NOT_USE_FOR_INFERENCE`.**

Its repair code used raw digit inputs `Xt` where the replacement was defined on internal activation `ta[0]`. Equal width 64 made the semantic error silent.

Consequences:

- Phase 2E `0/8` is not scientific negative evidence;
- Phase 2I's RNG causal explanation is retracted;
- 2H/2J interpretations tied to 2E are weakened/confounded;
- Phase 2O did not confirm a reliable composed repair-sample advantage (`UNCERTAIN`).

Invalidation history is preserved in:

- `results/phase2/precision_composition/CORRECTION_STATUS.json`
- `results/phase2/precision_composition/INVALIDATED_HISTORY.md`

Not all later 2D–2O raw per-seed artifacts are checked into this Git branch; the correction archive is identified by SHA256 `1a339be12d7644de534ac77a712307c49ee0c3d9acb28c8a3532883edca3dab7`.

## Reproducibility / systems boundary

- G7 seed 4300 exact portable rerun: retained as reproduction/portability evidence, not independent replication.
- Runtime PoC: retained only as one small CPU/storage/inference result.
- Meaningful host-RAM reduction, GPU/VRAM/energy/large-model/general runtime gains: not established.

## Publication gate checks

Issue #9 may close only when the final reviewed branch satisfies:

1. public claim corrections are present;
2. invalidation history is preserved;
3. minimal public residual-MLP runner smoke test passes;
4. `python tools/audit_repo.py` passes;
5. GitHub `repository-audit` passes for the final reviewed commit.

`docs/RELEASE_CHECKLIST.md` tracks these checks.

## Stopping rule

Before public release, continue only:

- correction/provenance work;
- minimal reproducibility smoke checks;
- CI/audit fixes;
- release/version-control metadata and PR review.

Do **not** open a broad new experimental family merely to make the repository feel complete.
