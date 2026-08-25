# Core discovery: compositional simplification of learned computation

## Short statement

The central empirical observation in Canaria is:

> **A sequence of learned computations can sometimes be simpler to replace as one composed function than its implementation-level components are to replace separately.**

This document defines that claim narrowly enough to be testable and distinguishes it from stronger claims that the experiments do not establish.

## Operational meaning of "simpler"

Canaria does not measure mathematical Kolmogorov complexity.

In this repository, simplification is operational and task-conditioned. Depending on the experiment, it is measured by one or more of:

- smaller replacement parameter count;
- smaller serialized/codec representation;
- lower candidate-grammar description cost;
- fewer blocks / narrower operators;
- task-preserving replacement under a declared fidelity or utility criterion.

The statement is therefore about **replacement/description complexity under an explicit experimental grammar and task distribution**.

## Evidence chain

### 1. Canary is not the phenomenon itself

Early work used a Canary signal as a possible local indicator of replaceability. Blinded evidence later showed that strong simplification also occurred in low-Canary regions. The low-Canary strong-simplification rate was 0.845 with a 95% seed-cluster CI of 0.7225–0.9500 in the original tested setting.

Conclusion: Canary is at most a partial observer of boundary/contract stress; it is not a necessary local condition.

### 2. Implementation boundaries are not always functional boundaries

Individual implementation blocks can be poor simplification units. Expanding the replacement boundary or merging adjacent spans can expose a simpler input-output mapping.

This is conceptually important: the complexity assigned to an implementation component need not equal the complexity of the function realized by a larger composed span on the task distribution.

### 3. Composition subadditivity was common in the original confirmatory setting

The original confirmatory composition test reported:

- `P(G > 0) = 0.7107`
- 95% CI `0.6128–0.8137`

where positive `G` denotes a composition gain under the declared candidate grammar.

This supports the statement that composition complexity was frequently subadditive in that tested setting.

It does **not** establish that all neural-function composition is subadditive or that the result is grammar-independent.

### 4. Direct cross-family replication in a Small Vision Transformer

A later fresh confirmatory experiment tested the same core question directly in a different architecture family.

Teacher:

- 4-block SmallViT on sklearn digits;
- fixed central two-block span, blocks 1–2.

Common replacement grammar:

- identity; or
- one Transformer block with hidden dimension 32, four heads, and MLP width in `{8,16,32,64}`.

Two strategies were compared under the same locked passing rule (`held-out span NMSE <= 0.12` and validation utility `>= 0.95`):

- **component-wise:** replace the two source blocks separately;
- **composed:** fit the two-block input-output map directly with one replacement operator.

Across the first 8 fresh baseline-eligible seeds `>=9000`:

- component-wise minimum passing complexity: **9,808 replacement parameters** in 8/8 seeds;
- composed minimum passing complexity: **4,904–5,424 parameters**;
- mean composed/component-wise complexity ratio: **0.51988**;
- paired seed-bootstrap 95% CI: **[0.50634, 0.53926]**;
- composed smaller: **8/8 seeds**;
- selected composed mean test utility, on data not used for selection: **0.97856**, bootstrap95 **[0.97090, 0.98562]**.

The component-wise strategy used 640 compiler updates while the composed strategy used 320, so the composed complexity advantage was not purchased with greater fit effort.

See `CROSS_FAMILY_COMPOSITION_REPLICATION.md` and `results/replication/vit_compositional/`.

### 5. Second direct architecture-family replication in a residual MLP

A second fresh confirmatory experiment tested the same operational question in a four-block residual MLP on sklearn digits.

The fixed span was the first two residual blocks. The replacement grammar used bias-free residual bottleneck modules:

- **component-wise:** two width-`h` replacement modules;
- **composed:** one width-`2h` replacement module.

For every budget grid point the learned replacement-parameter count was **exactly matched**:

- component-wise total: `256h`;
- composed total: `256h`.

Fit effort was also conservatively matched by parameter-update count and approximate linear-layer compute. Budget selection used validation only:

- validation span NMSE `<= 0.08`;
- validation accuracy within 2 absolute percentage points of the teacher;
- select the smallest passing budget from `512,1024,1536,2048,3072,4096,6144`.

Fresh seeds `1200–1207` produced:

- component-wise mean minimum passing budget: **3584 params**;
- composed mean minimum passing budget: **1728 params**;
- composed smaller: **8/8 fresh seeds**;
- mean `log2(B_composed/B_componentwise) = -1.0519`;
- paired bootstrap95 **[-1.2075, -0.8962]**;
- geometric mean budget ratio: **0.4823×**;
- mean untouched-test accuracy difference at validation-selected budgets: **+0.00583** for composed minus component-wise, bootstrap95 **[+0.00306,+0.00806]**.

A preregistered mechanistic secondary at fixed 2048 replacement parameters compared local component targets with a joint span objective while keeping the two-module factorized topology:

- local component-wise NMSE: **0.1474**;
- same two-module architecture jointly fit end-to-end to the span target: **0.0639**;
- one composed module: **0.0533**.

This control is especially informative: most of the gap is recovered when the implementation still contains two replacement modules but the **optimization target is the composed span function**. The result therefore supports a functional-boundary/objective interpretation rather than attributing the entire effect to the one-module topology.

See `CORE_DISCOVERY_REPLICATION_DIGITS.md` and `results/core_discovery_digits/`.

### 6. Whole-network accounting matters

A local replacement can appear small only because complexity moves into the surrounding network. Canaria therefore added whole-network accounting.

In the residual-CNN endpoint, matched whole-network accounting retained a real reduction under the declared representations, including approximately 26–29% reductions under fixed-FP32 / q8+zlib accounting and an independently confirmed exact 9,926-byte whole-network serialization endpoint.

This rejects the strongest form of the explanation that all observed local simplification was merely hidden complexity relocation.

### 7. Dynamic consolidation extends the static phenomenon

The later real-text LM experiments asked a different but related question: what happens when consolidation is performed during learning?

G15 showed that staged `4→3→2` consolidation with task learning between commits beat a direct `4→2` path. G17 then removed the intervening task learning while keeping a two-stage compiler path; that factorized path became equivalent to direct compilation.

The key inference is therefore not merely "two smaller fits are easier than one large fit." The intervening learning/recontracting phase matters.

G19 replicated the staged advantage on a different path (`5→4→2` versus `5→2`) with identical compiler-update budgets.

## Current interpretation

The most useful current interpretation is:

1. neural implementation boundaries can overstate the task-effective complexity of the function that crosses them;
2. composing a wider span can expose cancellation, redundancy, low-dimensional task manifolds, or other structure that is not visible under component-wise accounting;
3. direct SmallViT and residual-MLP fresh replications show that this operational effect is not confined to the original residual-CNN implementation family;
4. the residual-MLP joint-factorized control indicates that much of the gain follows the **composed functional objective itself**, not merely the use of a one-module replacement topology;
5. after a consolidation is committed, continued task learning reorganizes the remaining computation;
6. that reorganization can make the next compiler easier to optimize while also making the task more sensitive to residual approximation error.

The sixth point is important: **"easier to fit" is not the same as "safer to approximate."**

## Strong claims to avoid

The repository does not establish:

- that function composition always lowers complexity;
- that the mathematical function itself has lower Kolmogorov complexity;
- that one candidate grammar measures an intrinsic universal description length;
- that every learned network contains large compositional simplifications;
- that any particular Canary metric is necessary or sufficient;
- that parameter reduction automatically implies wall-clock or energy reduction;
- that the SmallViT or residual-MLP results imply universal Transformer/LLM or task-universal behavior.

## Suggested research wording

A strong but defensible summary is:

> **We identify and characterize an empirical phenomenon of task-conditioned compositional simplification in learned neural computation: computations that are difficult or expensive to simplify component-wise can sometimes admit a substantially simpler task-preserving representation when treated as a single composed function. The phenomenon was observed in the original residual-CNN setting and directly replicated under locked fresh protocols in both a Small Vision Transformer and a residual MLP.**

A dynamic extension supported by the training-time experiments is:

> **After consolidation, continued task learning can reorganize the remaining computation in ways that change the difficulty and task sensitivity of subsequent consolidations.**

## Why this matters

The main value of the observation is not another pruning recipe. It changes the unit at which neural computation is analyzed.

Instead of assuming:

```text
network complexity ≈ sum of implementation-component complexities
```

Canaria asks whether a more useful description is:

```text
find functional boundaries
→ compose learned computation
→ search for a simpler task-conditioned realization
→ optionally resume learning and repeat
```

That viewpoint motivates both the scientific questions in this repository and the possible runtime/compiler applications in `APPLICATIONS.md`.
