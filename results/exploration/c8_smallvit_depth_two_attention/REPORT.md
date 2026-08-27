# C8 SmallViT depth-two hierarchy with attention-aware Canaria grammar — exploratory report

Decision: **EXPLORATORY GRAMMAR LIMITATION**.

C8 introduced a bias-free self-attention residual candidate whose learned parameter count scales exactly linearly with width, allowing exact aggregate budget matching across four local units, two recursively generated pair units, and one final depth-two unit.

Fresh exploratory seeds: `1390–1392`. All three teachers were eligible. Held-out test data was not evaluated.

## Exact budget

- Level 0: four `TinyAttnRes(32,16)` = **8192** aggregate learned parameters.
- Level 1: two `TinyAttnRes(32,32)` = **8192** aggregate learned parameters.
- Level 2: one `TinyAttnRes(32,64)` = **8192** learned parameters.

## Relative hierarchy behavior

Level-2 joint adaptation using only the lower-level Canaria IR improved final functional NMSE over no Level-2 adaptation in all three seeds:

- 1390: `-0.04516`
- 1391: `-0.01964`
- 1392: `-0.03356`

Depth-two / flat-lower-IR NMSE ratios were `1.0264x`, `0.9880x`, and `1.0463x`. Thus there is no sign here that the extra hierarchy level itself causes uncontrolled error growth.

## Absolute grammar failure

However, the attention-only grammar is not adequate for the full four-block SmallViT span.

Mean validation NMSE versus the original teacher:

| representation | mean NMSE |
|---|---:|
| lower-level composed Canaria IR | 0.44919 |
| depth-two, no Level-2 adaptation | 0.70434 |
| depth-two + Level-2 joint adaptation | 0.67156 |
| flat single from same lower IR | 0.65945 |
| direct attention-only single from original | 0.57766 |

Mean replacement validation accuracy:

- depth-two joint: **0.2420**
- flat lower-IR single: **0.2494**
- direct original single: **0.3333**

These are far below the eligible SmallViT teacher accuracies (~0.956–0.970). Seed 1390 was effectively at chance level for all recursive/flat replacements.

## Interpretation

C8 therefore must **not** be cited as positive cross-family depth-two evidence. Its correct role is a grammar limitation result:

- self-attention token mixing alone is insufficient to represent four SmallViT blocks;
- the hierarchy mechanism remains comparatively stable relative to a flat compile from the same poor lower IR;
- but the lower IR itself is too inaccurate for a scientifically useful cross-family hierarchy claim.

The next experiment should add feed-forward nonlinearity while preserving the exact linear-in-width parameter accounting. A bias-free attention + MLP residual grammar satisfies that requirement and is a cleaner test of whether C8 failed because the candidate family was under-expressive.
