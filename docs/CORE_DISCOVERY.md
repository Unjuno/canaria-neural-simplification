# Core discovery: task-conditioned compositional simplification

## Short statement

The central empirical observation in Canaria is:

> **Under an explicit task distribution, replacement grammar, and passing rule, a sequence of learned computations can sometimes admit a smaller task-preserving replacement when fitted as one composed input-output function than when its implementation-level components are fitted separately.**

This is an **operational replacement/description-complexity** statement. Canaria does not measure Kolmogorov complexity or an intrinsic mathematical complexity of the learned function.

## What “simpler” means here

Depending on the experiment, “simpler” can mean:

- fewer learned replacement parameters;
- fewer replacement blocks / narrower operators;
- smaller declared coded or serialized representation;
- lower candidate-grammar description cost;
- passing a declared functional-fidelity and task-utility contract at lower replacement cost.

Every claim is therefore conditional on its grammar, accounting rule, span, data distribution, and endpoint criterion.

## Evidence chain

### 1. Canary is not the phenomenon itself

The original blinded evidence found strong simplification in low-Canary regions. High Canary is not a necessary local condition under the tested sensor definition.

### 2. Implementation boundaries are not universally privileged functional boundaries

Wider-span and merged replacements can succeed where local component replacement is weaker. This motivates measuring the task-conditioned input-output span rather than assuming implementation blocks are the correct simplification atoms.

### 3. Original operational composition-gain result

The original confirmatory program reported:

```text
P(G > 0) = 0.7107
95% CI    = [0.6128, 0.8137]
```

Positive `G` means positive **composition gain under the declared candidate grammar**.

Publication-safe wording is therefore:

> positive operational composition gain was frequent in the original tested setting.

Do **not** restate this as an intrinsic law that “composition complexity is subadditive.”

### 4. Fresh SmallViT direct replication

A fixed central two-block span in a four-block SmallViT on `sklearn` digits was tested under a locked component-wise-versus-composed replacement protocol.

Fresh eligible seeds:

`9000, 9003, 9004, 9007, 9008, 9009, 9010, 9011`

Results:

- component-wise selected replacement: `9,808` parameters in 8/8;
- composed selected replacement: `4,904–5,424` parameters;
- mean composed/component-wise ratio: `0.51988`;
- bootstrap95 ratio: `[0.50634, 0.53926]`;
- composed smaller: `8/8`;
- selected composed mean test utility: `0.97856`.

Important audit boundary: the locked selection rule excludes test accuracy, but the runner records test metrics for every candidate. Test was therefore not a selection variable, but it was not operationally hidden during candidate-result generation. The primary result remains usable under the locked protocol; future runners should delay test evaluation until after selection.

See `CROSS_FAMILY_COMPOSITION_REPLICATION.md`.

### 5. Fresh residual-MLP direct replication with exact learned-budget matching

The stronger matched-budget control used a four-block residual MLP on `sklearn` digits, fixed span = first two residual blocks.

At every budget grid point:

```text
component-wise total learned replacement params = 256 h
composed total learned replacement params       = 256 h
```

Fresh seeds `1200–1207`:

- component-wise mean minimum passing budget: `3584`;
- composed mean minimum passing budget: `1728`;
- composed smaller: `8/8`;
- mean `log2(B_composed/B_componentwise) = -1.0519`;
- bootstrap95 `[-1.2075,-0.8962]`;
- geometric mean budget ratio `0.4823×`;
- selected-budget test-accuracy difference `+0.00583`, bootstrap95 `[+0.00306,+0.00806]`.

Selection uses validation NMSE/accuracy; test evaluation follows minimum passing budget selection.

### 6. Joint-factorized secondary control

At fixed `2048` learned replacement parameters:

- local component-wise NMSE: `0.1474`;
- same two-module topology jointly fitted to the span target: `0.0639`;
- one composed module: `0.0533`.

This was a preregistered **descriptive/mechanistic secondary with no confirmatory pass rule**. It is consistent with much of the local gap following the composed span objective rather than only the one-module topology. It does not establish a causal decomposition theorem.

See `CORE_DISCOVERY_REPLICATION_DIGITS.md`.

### 7. Whole-network accounting

The original residual-CNN work also added whole-network accounting to test whether apparent local savings were simply moved elsewhere. Under its declared codecs, a material whole-network reduction remained. This rejects the strongest hidden-relocation explanation **under those accounting schemes**, not for all possible codecs.

### 8. Dynamic extension

Training-time consolidation asks a related but distinct question: what happens after a compact replacement is committed and task learning resumes?

The re-reviewed boundary is:

- G7 primary: progressive consolidation beat preregistered early/late one-shot controls;
- G15: staged `4→3→2` with intervening task learning beat the tested direct-wait path;
- G17: back-to-back factorized fitting without intervening learning was equivalent to direct contraction;
- G19: staged `5→4→2` beat direct `5→2` under equal compiler-update counts.

These results support a learning-path/recontracting effect in the small character-LM testbed. They do not prove a universal consolidation algorithm.

## Candidate mechanisms versus established facts

Possible explanations for why a wider functional span can be easier to replace include cancellation, redundancy, restricted task manifolds, representation redistribution, and optimization-objective effects.

These are **mechanistic hypotheses unless directly controlled**. The residual-MLP joint-factorized secondary provides evidence that objective/boundary choice matters, but it does not uniquely identify the underlying mechanism.

## Strong claims to avoid

The repository does not establish:

- that function composition always lowers complexity;
- that mathematical or Kolmogorov complexity was measured;
- that one candidate grammar measures an intrinsic universal description length;
- that every network/span contains large compositional simplifications;
- that Canary is necessary or sufficient;
- that parameter reduction automatically yields wall-clock, energy, RAM, or GPU gains;
- that the SmallViT/residual-MLP results imply universal Transformer, LLM, or task-universal behavior.

## Publication-safe summary

> **Canaria identifies and characterizes an operational phenomenon of task-conditioned compositional simplification: under explicit replacement grammars and task-preservation criteria, some learned spans admit smaller replacements when fitted as composed functions than when simplified at implementation-component boundaries. The effect was observed in the original residual-CNN program and directly tested under fresh locked protocols in a Small Vision Transformer and a residual MLP.**

See `CLAIMS_AND_EVIDENCE.md` for the authoritative current claim registry and `INDEPENDENT_REREVIEW_2026-08-26.md` for the pre-publication decision ledger.
