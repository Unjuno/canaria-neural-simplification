# Publication / communication notes

This document defines publication-safe wording after the 2026-08-26 independent re-review. It is not a manuscript.

## One-sentence project statement

> Canaria identifies an empirical pattern of **task-conditioned compositional simplification**: under explicit task distributions, replacement grammars, and passing criteria, some learned spans admit smaller task-preserving replacements when fitted as one composed input-output function than when simplified at implementation-component boundaries.

## Primary discovery claim

Use **operational replacement/description complexity**, not intrinsic mathematical complexity.

Safe:

- “Under the declared grammar, positive operational composition gain was frequent in the original confirmatory setting.”
- “The component-wise-versus-composed effect was tested under fresh locked protocols in a Small Vision Transformer and a residual MLP.”
- “The residual-MLP experiment exactly matched learned replacement-parameter budgets at every grid point.”

Avoid:

- “Function composition reduces complexity.”
- “Composition complexity is intrinsically subadditive.”
- “We measured Kolmogorov complexity.”
- “The SmallViT/residual-MLP results prove universal Transformer or LLM behavior.”

## Direct SmallViT replication

Public result:

- 8/8 fresh eligible seeds selected a smaller composed replacement;
- mean composed/component-wise replacement-parameter ratio `0.51988`;
- paired seed-bootstrap95 `[0.50634, 0.53926]`;
- selected composed mean test utility `0.97856`.

Important isolation caveat: the locked selection criterion excludes test accuracy, but the runner records test accuracy for every candidate. Therefore say **“test was not a selection variable”**, not “the test set was operationally hidden until after selection.”

Do not use SmallViT as an exact parameter-matched mechanistic decomposition; the residual MLP provides the stronger matched-budget control.

## Direct residual-MLP replication

Fresh `1200–1207`:

- component-wise mean minimum passing budget: `3584`;
- composed mean minimum passing budget: `1728`;
- composed lower: `8/8`;
- mean `log2(B_composed/B_componentwise)=-1.0519`;
- bootstrap95 `[-1.2075,-0.8962]`;
- geometric mean ratio `0.4823×`;
- selected-budget test-accuracy difference `+0.583` percentage points, bootstrap95 `[+0.306,+0.806]` pt.

Candidate selection used validation only, and learned replacement-parameter counts were exactly matched at each budget.

The 2048-parameter joint-factorized control is **descriptive/mechanistic secondary**. It is consistent with much of the gap following the composed span objective, but it is not a confirmatory causal decomposition.

## Training-time consolidation

Keep evidence tiers explicit.

### G7

The preregistered primary PASS comparisons were progressive versus **early one-shot** and **late one-shot**. Those passed.

The small-from-start and large-reference differences are useful **secondary** observations. Do not present “progressive beats small-from-start” as if it were the G7 primary decision rule.

### G15 / G17 / G19

Safe synthesis:

> Under the tested small character-LM protocols, staged consolidation with intervening task learning beat the tested direct path, while back-to-back factorized fitting without intervening learning did not reproduce the staged advantage.

### G18

Safe:

> The tested deadline-aware controller improved mean PPL and reduced mean compiler updates relative to the tested static NMSE controller.

Avoid turning this into a universal rule that remaining horizon must determine commit timing.

### G20–G26

Safe synthesis:

> Recontracting can make a subsequent compiler easier to optimize at a fixed normalized error target while making residual error more task-sensitive; sensitivity-aware empirical predictors improved task-damage prediction in the tested protocols.

Do not present empirical predictor coefficients as a theorem or universal Taylor law.

## Phase 2 correction

### Valid A–C boundary

- 2A: 4-bit composed coded-size result — **VALID_PASS** under the declared residual-MLP quantizer/accounting.
- 2B: capacity-only rescue of naive 3-bit per-matrix PTQ — **VALID_FAIL**.
- 2C: row-wise scale rescue — **VALID_PASS**, and it rescues both topologies.

### Phase 2E

Phase 2E is **INVALIDATED_IMPLEMENTATION_BUG**, not negative scientific evidence. Repair used raw `Xt` where the replacement was defined on internal activation `ta[0]`; width 64 made the error silent.

The `0/8` result must not be cited as evidence of stochastic repair failure. Preserve it only as correction history.

### Later corrected boundary

Corrected later work supports viability of short activation-domain QAT-style repair for coarse per-matrix 3-bit in the tested residual-MLP family.

Do **not** claim lower repair-sample complexity for composition: Phase 2O is `UNCERTAIN` (`p=0.1662`; bootstrap95 crosses zero).

Not all later 2D–2O raw per-seed artifacts are checked into this branch. The later correction archive is identified by SHA256 `1a339be12d7644de534ac77a712307c49ee0c3d9acb28c8a3532883edca3dab7`.

## Reproducibility wording

A self-contained public runner reproduced the archived G7 seed-4300 output exactly in the recorded environment, with SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

Call this **reproduction/portability evidence for an already-confirmatory seed**, not an independent scientific replication.

## Systems wording

Publication-safe PoC statement:

> In one small CPU proof of concept, a progressively consolidated learned representation was about 50% smaller as a serialized artifact and about 2× faster for the measured batch-128 inference workload, while meaningful host-RAM reduction was not demonstrated.

Do not generalize to GPU, VRAM, energy, cold-start, browser/edge, large models, or universal runtime speedup.

## Required negative/correction boundaries to mention

A credible public account should include at least:

- Canary is not necessary.
- Unlimited recursive collapse is not supported.
- Teacher-forced PPL can remain close while autoregressive rollouts diverge.
- G17 rejects “two compiler fits alone” as the staged-gain explanation under the tested protocol.
- G21 hard task-damage veto can block contraction.
- G27 fixed risk caps did not establish a Pareto improvement.
- Phase 2E is invalidated and excluded from inference.
- Phase 2O does not confirm a composed repair-sample advantage.
- The runtime PoC does not demonstrate meaningful host-RAM reduction.

## Publication state

Do not describe the research branch as publication-ready until the independent review gate is actually closed and the final `repository-audit` passes. Closing Issue #9 completes this quality gate only; PR #7 and the v0.2.0 release/tag boundary remain separate version-control decisions.

See `INDEPENDENT_REREVIEW_2026-08-26.md` and `RELEASE_CHECKLIST.md`.

## Novelty wording

The repository supports saying that Canaria **identifies and characterizes** this empirical phenomenon and directly tests it in additional small architecture families. A strict “first ever” priority claim still requires a dedicated literature review.

## Citation practice

Until a paper exists, cite the exact commit/tag, the relevant protocol/result artifacts, and the evidence class. Never cite an invalidated artifact as scientific support merely because it remains in the history.
