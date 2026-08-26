# Direct cross-family replication of compositional simplification

## Purpose

This experiment tests the core operational phenomenon in a Small Vision Transformer on `sklearn` digits:

> Under a declared replacement grammar and task/fidelity rule, does fitting a fixed two-block input-output map directly require fewer replacement parameters than fitting the two blocks separately?

This is a direct architecture-family replication of the component-wise-versus-composed question. It is not evidence of universal Transformer or LLM behavior.

## Locked design

Teacher:

- 4 Transformer blocks;
- hidden dimension 32;
- 4 attention heads;
- MLP width 64;
- fixed central span: blocks 1 and 2.

Replacement operators are either identity or one Transformer block with hidden dimension 32, four heads, and MLP width in `{8,16,32,64}`.

Strategies:

- **component-wise:** fit block 1 and block 2 separately to teacher intermediate targets and compose the two replacements at execution time;
- **composed:** fit the two-block input-output map directly with one replacement operator.

Each non-identity compiler receives 40 fit epochs. A selected component-wise pair therefore uses twice the compiler updates of a selected one-operator composed candidate; optimization effort does not favor the composed strategy.

Passing rule:

- training-held-out span NMSE `<= 0.12`;
- validation utility `>= 0.95` relative to the baseline teacher.

Within each strategy, select the passing candidate with the fewest replacement parameters, tie-breaking by lower held-out NMSE.

Fresh confirmation uses the first 8 baseline-eligible seeds `>=9000`; eligibility is validation accuracy `>=0.95`. Exploratory 8900-series seeds are excluded.

Protocol: `results/replication/vit_compositional/PROTOCOL_LOCK.json`.

## Test-set isolation boundary

The **locked selection rule does not use test accuracy**. However, the public runner evaluates and records test accuracy for every candidate, not only after the selected candidate is fixed.

Therefore the correct statement is:

> test metrics are excluded from the preregistered selection rule, but the test split was not operationally hidden during candidate-result generation.

Because the protocol, code hash, fresh-seed rule, and deterministic selection criterion were locked before fresh outcomes, the primary minimum-passing-complexity result remains usable. This is nevertheless weaker isolation practice than the residual-MLP runner, which delays test evaluation until the minimum passing budget has been selected. Future replication runners should follow the latter pattern.

## Confirmatory result

Eligible fresh seeds:

`9000, 9003, 9004, 9007, 9008, 9009, 9010, 9011`

Across all 8:

- selected component-wise replacement: **9,808 parameters**;
- selected composed replacement: **4,904–5,424 parameters**;
- mean composed/component-wise replacement-parameter ratio: **0.51988**;
- geometric mean ratio: **0.51926**;
- paired seed-bootstrap 95% CI: **[0.50634, 0.53926]**;
- composed selected representation smaller: **8/8**.

Primary decision: **PASS under the locked operational rule**.

Selected composed mean test utility was **0.97856** (bootstrap95 **[0.97090, 0.98562]**); selected component-wise mean test utility was **0.98479**. Treat these test metrics as generalization observations, not as selection variables.

## Interpretation

The supported statement is operational and grammar-dependent:

> In this SmallViT task, span, replacement family, and passing rule, the directly fitted two-block map admitted a smaller passing replacement than component-wise treatment in all eight fresh eligible seeds.

The result does **not** establish:

- universal subadditivity for Transformer spans;
- mathematical or Kolmogorov complexity reduction;
- large-LLM generalization;
- an exact parameter-matched mechanistic decomposition comparable to the residual-MLP control;
- that the one-operator topology itself is the sole cause of the observed gap;
- that the exact ~0.52 ratio is universal.

## Evidence files

- `results/replication/vit_compositional/PROTOCOL_LOCK.json`
- `results/replication/vit_compositional/confirm_summary.json`
- `results/replication/vit_compositional/seed_table.csv`
- `scripts/replication/vit_compositional.py`

## H / T / D / C / U

**H** — The selected minimum-passing replacement parameter count is lower for the directly composed two-block map than for separate component-wise replacement.

**T** — SmallViT, fixed central span, locked grammar/thresholds, first-8-eligible fresh-seed rule. Selection excludes test metrics, although the runner records test values for every candidate.

**D** — **PASS**: composed smaller in 8/8; bootstrap95 upper ratio `0.539 < 1`; mean selected composed test utility `0.979`.

**C** — The effect may depend on task manifold, compiler grammar, span location, topology, and model scale. Intermediate-target distribution mismatch is part of the operational component-wise strategy.

**U** — Small `n=8`; external validity and stronger test-isolation replication remain open.
