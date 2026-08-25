# Research roadmap — frozen snapshot / future-work handoff

**Current state:** the v0.2.0 scientific snapshot is closed at its present claim scope. Broad experiment expansion is stopped.

The earlier cross-architecture roadmap is preserved in `GENERALIZATION_ROADMAP.md` and `GENERALIZATION_STATUS.md` as historical planning/evidence. It should not be read as a current commitment to run every listed experiment.

## Completed closure — clean-repository reproduction

A portable runner reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` assumptions. The generated JSON exactly matched the archived confirmatory seed output in the recorded environment, including SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

## Completed closure — minimal runtime materialization PoC

A bounded CPU-only systems PoC implements:

```text
trained large / compact model
→ serialize state + manifest
→ fresh-process load/materialize
→ direct execution
```

For G7 seed 4300:

- serialized artifact + manifest: **110,093 → 54,646 bytes** (**−50.36%**);
- parameters: **23,138 → 11,042** (**−52.28%**);
- mean CPU batch-128 inference: **47.05 → 23.11 ms**;
- RSS delta: **4.72 → 4.56 MB**, so meaningful host-RAM reduction was not demonstrated.

## Completed closure — direct cross-family replication of the core discovery

The core component-wise-versus-composed effect now has two direct fresh architecture-family replications beyond the original residual-CNN program.

### Small Vision Transformer

Fresh confirmatory result:

- component-wise minimum passing complexity: **9,808 replacement params** in all 8 seeds;
- composed minimum passing complexity: **4,904–5,424 params**;
- mean composed/component-wise ratio: **0.51988**;
- paired bootstrap95: **[0.50634, 0.53926]**;
- composed smaller: **8/8 seeds**;
- selected composed mean test utility: **0.97856**;
- compiler updates: **640 component-wise vs 320 composed**.

See `CROSS_FAMILY_COMPOSITION_REPLICATION.md` and `results/replication/vit_compositional/`.

### Residual MLP

A second fresh confirmatory experiment on a four-block residual MLP used exact learned-parameter-budget matching between component-wise and composed replacement.

Fresh seeds `1200–1207`:

- component-wise mean minimum passing budget: **3584 params**;
- composed mean minimum passing budget: **1728 params**;
- composed lower-budget: **8/8 seeds**;
- mean `log2(B_composed/B_componentwise) = -1.0519`;
- bootstrap95 **[-1.2075,-0.8962]**;
- geometric mean budget ratio **0.4823×**;
- untouched-test accuracy difference at validation-selected budgets: **+0.583 pt**, bootstrap95 **[+0.306,+0.806] pt**.

At fixed 2048 params, local component-wise span NMSE was **0.1474**, the same two-module architecture jointly optimized on the span target reached **0.0639**, and one composed module reached **0.0533**. This indicates that much of the effect follows the composed **functional objective/boundary** rather than only the one-module topology.

See `CORE_DISCOVERY_REPLICATION_DIGITS.md` and `results/core_discovery_digits/`.

## Current stopping rule

All bounded closure tasks for this public snapshot are complete. Canaria should now be treated as a **frozen research snapshot** at the current claim scope.

Do not continue the old G-number sequence by default. New scientific work should start from a new issue/research phase with its own claim, protocol, and stopping rule.

## Future research topics

These remain scientifically interesting but are not required for closure:

- larger pretrained Transformer/LLM external validity;
- replication on additional **task types** rather than only additional architecture families;
- additional spans, widths, and replacement grammars;
- codec-independent complexity/MDL;
- off-manifold functional complexity;
- stronger null models and known-complexity synthetic teachers;
- effective repair/tangent dimension;
- mechanism algebra/dictionaries;
- sensitivity-aware utility-cost controllers;
- hardware-specific functional IR and JIT execution beyond the minimal PoC;
- larger-scale memory/energy/runtime benchmarking.

See `OPEN_QUESTIONS.md` for the handoff list.
