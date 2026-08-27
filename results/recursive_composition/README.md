# Recursive-composition results

Use `docs/RECURSIVE_COMPOSITION.md` for the evidence map and claim boundaries.

## Confirmatory

Current confirmed stages:
- `c3_recursive_recompilation/`
- `c5_smallvit_recursive/`
- `c7_depth2_recursive/`
- `c9_depth3_realignment/`
- `c13_half_interface/`
- `c15_basis_robustness/`
- `c17_quarter_interface/`
- `c20_smallvit_half_interface/`
- `c21_smallvit_quarter_interface/`
- `c23_anchor_identity/`
- `c25_smallvit_anchor_identity/`

The later confirmatory directories contain locked protocols, machine-derived seed rows, preregistered aggregate results, reports, and independent bootstrap audits. C20 was additionally reproduced from the byte-identical committed runner in the local container before the new C24/C25 work.

## Exploration and negative evidence

The `exploration/` subtree preserves C1, C2, C4, C6, C8, C10, C11, C12, C14, C16, C18, C19, C22 and C24 records. Exploratory outcomes are calibration/mechanism evidence and must not be promoted to confirmatory claims.

Notable negative/informative evidence is retained:
- C11: naive partial hidden sketches fail without complement anchoring;
- C18: full-four-block SmallViT token-wise replacement collapses task utility to chance even though relative self-anchor ordering remains visible;
- C22/C24: generic complement anchors do not reproduce the sample-specific pre-Canaria self-anchor effect in the tested residual-MLP and SmallViT regimes; these explorations motivated C23/C25.
