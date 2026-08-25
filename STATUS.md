# Project status

**2026-08-25: release-ready, scientifically closed public snapshot at the current claim scope.**

Broad experiment expansion is stopped for this snapshot. The representative public reproduction, bounded runtime/materialization PoC, and direct cross-family replications of the core compositional-simplification phenomenon are complete. **No additional experiment is required for the present scoped claims.**

## Current project-level thesis

The central result is **task-conditioned compositional simplification of learned neural computation**:

> implementation-level components that are difficult or expensive to simplify separately can sometimes admit a substantially simpler task-preserving representation when treated as one composed input-output function.

The static phenomenon is now supported by:

- the original residual-CNN confirmatory composition evidence;
- a fresh Small Vision Transformer component-wise-versus-composed replication; and
- a second fresh residual-MLP component-wise-versus-composed replication with exact learned-parameter-budget matching.

The dynamic extension is:

> **form → transfer → commit → recontract → transfer again**

Intervening task learning after a structural consolidation changes the subsequent optimization geometry. In the current small real-text LM testbed, later compiler fitting becomes easier in normalized functional-error terms, while downstream sensitivity to residual error increases.

## Current evidence frontier

### Confirmed

- original composition subadditivity (`P(G>0)=0.7107`, 95% CI `0.6128–0.8137`) under the declared grammar;
- direct SmallViT replication: composed/component-wise minimum passing replacement-complexity ratio **0.5199**, paired bootstrap95 **[0.5063, 0.5393]**, composed smaller in **8/8 fresh seeds**, selected composed mean test utility **0.9786**;
- direct residual-MLP replication: component-wise mean minimum passing budget **3584 params** vs composed **1728 params**, composed lower in **8/8 fresh seeds**, mean `log2(B_comp/B_sep)=-1.0519`, bootstrap95 **[-1.2075,-0.8962]**, with test accuracy difference **+0.583 pt** bootstrap95 **[+0.306,+0.806] pt**;
- whole-network reductions under declared codecs, including an exact 9,926-byte residual-CNN endpoint;
- training-time staged consolidation (G7);
- function-aligned transfer requirement (G8);
- diminishing returns to transfer fit (G9);
- inheritance + functional refinement (G10);
- autonomous consolidation under a locked non-inferiority protocol (G11);
- staged-vs-direct path effect (G15);
- factorization-without-learning equivalence control (G17);
- deadline-aware controller improvement (G18);
- staged-path replication on `5→4→2` (G19);
- lower normalized next-compiler fit cost after recontracting (G20d);
- higher immediate task sensitivity at matched normalized error (G20e, G22);
- sensitivity-aware immediate-damage prediction (G23–G25);
- horizon-aware future-damage prediction (G26).

### Confirmed negative / boundary results

- Canary is not a necessary local condition for simplification.
- Teacher-forced PPL is not sufficient evidence of autoregressive functional equivalence.
- The tested v23–v25 natural-text post-hoc objectives did not recover rollout-sensitive fidelity.
- A hard task-damage veto (G21) can prevent final contraction and increase compiler cost.
- A single fixed risk cap did not produce a cost/utility Pareto improvement in G27 exploration.
- Unlimited recursive collapse is not supported by the current grammar.
- The SmallViT and residual-MLP replications do not imply universal Transformer/LLM or task-universal subadditivity.

## Cross-family core-discovery replications — complete

### Small Vision Transformer

A fresh confirmatory SmallViT experiment compared a fixed two-block span under a common replacement grammar.

Result:

- component-wise minimum passing complexity: **9,808 params** in all 8 seeds;
- composed minimum passing complexity: **4,904–5,424 params**;
- mean ratio: **0.51988**;
- paired bootstrap95: **[0.50634, 0.53926]**;
- composed smaller: **8/8**;
- composed mean test utility: **0.97856**, bootstrap95 **[0.97090, 0.98562]**;
- compiler updates: **640 component-wise vs 320 composed**.

See `docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md` and `results/replication/vit_compositional/`.

### Residual MLP

A second fresh confirmatory experiment used a four-block residual MLP on sklearn digits and simplified the first two residual blocks.

For every budget point, component-wise and composed conditions had **exactly the same learned replacement-parameter count**. Fit effort was matched by parameter-update count and approximate linear-layer compute. Candidate selection used validation NMSE/accuracy only; test accuracy was untouched until after budget selection.

Result across fresh seeds `1200–1207`:

- component-wise mean minimum passing budget: **3584 params**;
- composed mean minimum passing budget: **1728 params**;
- composed smaller: **8/8**;
- mean `log2(B_composed/B_componentwise)`: **−1.0519**;
- paired bootstrap95: **[−1.2075, −0.8962]**;
- geometric mean budget ratio: **0.4823×**;
- mean test-accuracy difference at validation-selected budgets: **+0.00583**, bootstrap95 **[+0.00306,+0.00806]**.

At fixed 2048 parameters, a mechanistic control found validation span NMSE:

- local component-wise: **0.1474**;
- same two-module architecture jointly fit to the composed span target: **0.0639**;
- one composed module: **0.0533**.

Most of the benefit is therefore recovered by changing the **functional objective/boundary** while keeping the two-module topology, strengthening the interpretation that implementation boundaries can overstate task-effective functional complexity.

See `docs/CORE_DISCOVERY_REPLICATION_DIGITS.md` and `results/core_discovery_digits/`.

## Reproducibility closure — complete

A portable public runner reproduces **G7 fresh confirmatory seed 4300** without private `/mnt/data` imports.

In the recorded environment, the complete reproduced JSON exactly matched the archived confirmatory output with SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

See `scripts/reproduce/g7_confirmatory/` and `results/reproduction/g7_seed4300_report.json`.

This is software/reproducibility evidence for an already-confirmatory seed, not a new independent scientific replication.

## Runtime/materialization PoC — complete at small-model scope

Recorded G7 seed-4300 CPU-only PoC:

- serialized artifact + manifest: **110,093 → 54,646 bytes** (**−50.36%**);
- parameters: **23,138 → 11,042** (**−52.28%**);
- CPU batch-128 inference: **47.05 → 23.11 ms mean**;
- load/materialize: **7.85 → 5.86 ms mean**, secondary due cache sensitivity;
- process RSS delta: **4.72 → 4.56 MB**, so meaningful host-RAM reduction was **not demonstrated**;
- test PPL: **19.2784 large vs 18.9322 compact**.

See `docs/RUNTIME_POC.md`.

## Release state

All bounded closure tasks are complete. The residual-MLP replication adds a second direct architecture-family confirmation beyond the already-complete SmallViT replication.

The repository should now be treated as a **frozen public research snapshot** at its current claim scope. Future work should start as a new issue/research phase rather than extending the old G-number mainline.

## Future research, not current closure work

Possible future projects include:

- larger pretrained Transformer/LLM external validity;
- replication across additional task types, spans, widths, and replacement grammars;
- codec-independent complexity/MDL;
- hardware-specific functional IR/JIT execution;
- larger-scale RAM/VRAM/energy/runtime benchmarks;
- stronger cost-aware autonomous control.

## Current public documentation

- `docs/PUBLIC_SNAPSHOT.md`
- `docs/CORE_DISCOVERY.md`
- `docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md`
- `docs/CORE_DISCOVERY_REPLICATION_DIGITS.md`
- `docs/CLAIMS_AND_EVIDENCE.md`
- `docs/PUBLICATION_NOTES.md`
- `docs/TRAINING_TIME_CONSOLIDATION.md`
- `docs/LATE_STAGE_FINDINGS.md`
- `docs/NEGATIVE_RESULTS.md`
- `docs/APPLICATIONS.md`
- `docs/RUNTIME_POC.md`
- `docs/REPRODUCIBILITY.md`
- `docs/TERMINOLOGY.md`
- `docs/FAQ.md`
- `docs/ROADMAP.md`
- `docs/OPEN_QUESTIONS.md`
- `docs/RELEASE_CHECKLIST.md`

The repository should be read as an auditable **public research snapshot**, not as a production-ready compression/runtime library.
