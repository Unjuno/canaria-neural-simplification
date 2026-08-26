# Project status

**Current mode: reviewed public baseline with isolated post-publication research.**

## Version and publication boundary

- Frozen v0.2.0 tag: `v0.2.0-public-snapshot`.
- Frozen baseline commit: `556dce21c7a5516a16780cb28d528d1ff3968e53`.
- GitHub release: `Canaria v0.2.0 — Public Research Snapshot`.
- Independent re-review (Issue #9): completed and closed on 2026-08-26.
- Reviewed post-snapshot release candidate (PR #7): squash-merged into `main`.
- `main` is the reviewed public baseline. New research may be isolated on research branches/draft PRs and does not change this baseline until separately reviewed and merged.

The frozen v0.2.0 tag is not rewritten by later corrections or experiments. Current `main` carries the independently reviewed post-snapshot precision/correction work.

## Core claim retained

The project-level thesis is **task-conditioned compositional simplification under explicit operational rules**:

> Some learned spans can admit smaller task-preserving replacements when fitted as one composed input-output function than when simplified at implementation-component boundaries.

The strongest fresh direct replication on the reviewed public baseline is the residual MLP with exact learned replacement-parameter matching:

- fresh seeds `1200–1207`;
- composed lower minimum passing budget in `8/8`;
- geometric composed/component-wise budget ratio `0.4823×`;
- validation selects the endpoint; test evaluation follows selection.

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

Not all later 2D–2O raw per-seed artifacts are checked into Git; the correction archive is identified by SHA256 `1a339be12d7644de534ac77a712307c49ee0c3d9acb28c8a3532883edca3dab7`.

## Reproducibility / systems boundary

- G7 seed 4300 exact portable rerun: reproduction/portability evidence, not independent replication.
- Runtime PoC: one small CPU/storage/inference result only.
- Meaningful host-RAM reduction, GPU/VRAM/energy/large-model/general runtime gains: not established.

## Repository / research-state boundary

The publication sequence is complete: independent review, frozen tag/release boundary, reviewed post-snapshot merge, and integrity audit.

Repository organization follows `REPOSITORY_LAYOUT.md`:

- `main` contains reviewed public material;
- unmerged research branches and draft PRs are isolated work-in-progress, not public-claim updates;
- maintenance changes should remain science-neutral and separate from fresh experimental outcomes;
- historical protocols/results are preserved rather than rewritten to match later interpretation.

`repository-audit` on `main` is the continuing integrity check.

## Stopping rule

Do not continue the old publication/closure sequence or broaden a completed protocol after seeing its outcomes. New scientific work should begin from a new issue/research phase with its own hypothesis, protocol/evidence class, inferential-unit policy, and stopping rule.
