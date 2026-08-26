# Independent pre-publication re-review — 2026-08-26

This is the independent quality-gate review required by `REVIEW_HANDOFF.md` and GitHub Issue #9. It is a claim audit, not a new research phase.

## Review rule

Every material public claim is classified as:

- **KEEP** — directly supported and stated within scope.
- **EDIT** — evidence is usable, but wording, evidence tier, scope, or provenance must be corrected.
- **REMOVE** — not adequately supported as a public claim.
- **INVALIDATE** — the experiment or inference is technically invalid; preserve provenance but do not use it as evidence.

## Decision ledger

| Area | Decision | Public boundary after re-review |
|---|---|---|
| Core task-conditioned compositional simplification | **KEEP** | Supported as an operational replacement/description result under declared tasks, spans, grammars, and passing criteria; not mathematical/Kolmogorov complexity. |
| Original `P(G>0)=0.7107` composition result | **EDIT** | Describe as frequent **positive operational composition gain under the original declared grammar**, not as an intrinsic law that “composition complexity is subadditive.” |
| Residual-MLP direct replication | **KEEP** | Fresh `1200–1207`; exact learned replacement-parameter matching at every budget; validation-only budget selection; composed lower in 8/8; primary bootstrap interval below zero. |
| Residual-MLP joint-factorized mechanism control | **EDIT** | Retain only as preregistered descriptive/mechanistic secondary. It is consistent with much of the gap following the composed span objective; it is not a confirmatory causal decomposition. |
| SmallViT direct replication | **KEEP + EDIT boundary** | Locked rule and fresh-seed policy support the minimum-passing replacement result. The runner records test metrics for every candidate even though test is not a selection variable, so test isolation is criterion-level rather than operationally hidden. Do not describe this as the same exact-budget control as the residual MLP. |
| G7 progressive training-time consolidation | **KEEP + EDIT tier** | The confirmatory primary comparison is progressive versus early/late one-shot. The small-from-start and large-reference comparisons are informative secondary results, not the preregistered primary decision. |
| G15/G17 staged-vs-direct mechanism separation | **KEEP** | Under the tested small character-LM protocol, staged consolidation with intervening task learning beat direct/wait, whereas back-to-back factorized fitting without intervening learning was equivalent to direct contraction. |
| G18 horizon-aware controller | **EDIT** | Keep the specific result: the tested deadline-aware controller improved mean PPL and reduced mean compiler updates versus the tested static controller. Do not elevate this to a universal rule that horizon must determine commit timing. |
| G20d/G20e and G22–G26 | **KEEP within testbed** | Retain as small-model mechanism/prediction results. Predictor coefficients are empirical, not a theorem. |
| G21 hard veto | **KEEP as negative evidence** | The intervention failed its target-reach criterion; all-run PPL is not capacity-matched. |
| G27 fixed risk cap | **KEEP as exploratory/no-claim** | No Pareto improvement was established. |
| Phase 2A 4-bit coded-size result | **KEEP** | Valid under the locked residual-MLP, symmetric signed-uniform, declared scale-metadata accounting. |
| Phase 2B 3-bit capacity-only rescue | **KEEP as FAIL** | Increasing weight count alone did not rescue naive 3-bit per-matrix PTQ in the tested range. |
| Phase 2C row-wise-scale rescue | **KEEP** | Row-wise scales rescued both topologies in 7/8; the rescue is not uniquely compositional. |
| Phase 2E | **INVALIDATE** | `INVALIDATED_IMPLEMENTATION_BUG`: repair used raw `Xt` instead of internal activation `ta[0]`; equal width 64 hid the semantic error. The 0/8 result must not support scientific inference. |
| Phase 2I causal explanation of 2E | **REMOVE / RETRACT** | Repair RNG cannot be claimed as the explanation of Phase 2E because the activation-domain bug changed as well. |
| Phase 2H / 2J interpretations tied to 2E | **EDIT** | Preserve numerical observations where applicable, but weaken/remove mechanism claims that used the invalid 2E comparison or bug-defined cohort. |
| Corrected short activation-domain QAT repair (2D/2L family) | **KEEP with provenance boundary** | Supports viability of short repair for coarse per-matrix 3-bit in the tested residual-MLP family. Later-phase raw per-seed artifacts are referenced by the correction archive hash but are not all checked into this Git branch, so this is not a public portable-reproduction claim. |
| Phase 2O composed repair-sample advantage | **REMOVE as positive claim** | Confirmatory result is **UNCERTAIN** (`p=0.1662`; bootstrap interval crosses zero). A reliable lower repair-sample complexity for composition is not established. |
| G7 portable seed-4300 rerun | **KEEP** | Exact recorded-environment reproduction of one already-confirmatory seed; software/portability evidence, not an independent scientific replication. |
| Runtime/materialization PoC | **KEEP within boundary** | One small CPU, one seed, batch-128 workload: smaller serialized artifact and lower measured CPU inference latency. No general RAM/VRAM/GPU/energy/LLM/cold-start claim. |

## Code audit

### Residual-MLP public runner

`run_confirmatory.py` keeps train/validation/test splits separate, fits replacements on training activations, selects the smallest passing budget from validation criteria, and only evaluates selected endpoints on test. Component-wise and composed learned replacement-parameter counts are exactly matched at each grid point. No private `/mnt/data` dependency was found.

**Decision: KEEP.**

### SmallViT replication runner

The locked candidate selection rule depends on training-held-out NMSE and validation utility, not test utility. However, the runner computes and stores test accuracy for every candidate before the final summary is formed. Because the protocol, seed rule, code hash, and deterministic selection rule were locked before fresh outcomes, this does not by itself invalidate the replication, but it is a weaker isolation practice than an operationally hidden test set.

**Decision: KEEP the primary result; EDIT the isolation wording.** Future runners should delay test evaluation until after candidate selection.

### Phase 2A–C public runners

The portable runners use internal activations (`train_acts[0]`, `train_acts[1]`, `train_acts[2]`) consistently for replacement fitting and evaluation. The public 2A–C paths contain no `/mnt/data` dependency. Quantizer accounting is symmetric signed-uniform with declared FP16 scale metadata; there is no zero-point metadata because the quantizer is symmetric. Row-wise scale counts are charged per output row.

**Decision: KEEP A–C.**

### Phase 2E

The known `Xt` versus `ta[0]` semantic-domain error is exactly the class of shape-coincidence bug the review procedure required checking. Width equality made the invalid computation executable.

**Decision: INVALIDATE and preserve provenance.**

## Evidence-preservation decision

Invalidated evidence is not to be deleted or silently rewritten. The authoritative machine-readable status remains:

`results/phase2/precision_composition/CORRECTION_STATUS.json`

A human-readable invalidation ledger is retained in:

`results/phase2/precision_composition/INVALIDATED_HISTORY.md`

The later Phase 2K–2O correction archive is identified by SHA256:

`1a339be12d7644de534ac77a712307c49ee0c3d9acb28c8a3532883edca3dab7`

The repository must not imply that all later-phase raw artifacts are present in Git when they are not.

## Publication gate

This re-review is complete only after the claim edits, invalidation ledger, public-runner smoke test, and `repository-audit` all pass on the review branch. Closing Issue #9 certifies this quality gate only; PR #7 and the v0.2.0 release/tag boundary remain separate repository-state decisions.
