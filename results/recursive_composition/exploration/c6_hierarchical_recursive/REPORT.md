# C6 two-level hierarchical recursive Canaria recompilation — exploratory report

Decision: **EXPLORATORY POSITIVE, WITH A CLEAR DEPTH PENALTY**.

C6 asks whether recursively generated Canaria units can themselves be composed and recursively recompiled at a second hierarchy level. Three fresh seeds (`1370–1372`) were run under an exact 4096-parameter representation budget at every hierarchy level. No held-out test outcome was evaluated.

## Hierarchy

```text
C1 -> C2       C3 -> C4
  |               |
  | joint pair adaptation
  v               v
 pair12          pair34
  |               |
  | cluster-only recursive compile
  v               v
 C12             C34
       \         /
        \       /
        C12 -> C34
             |
             | Level-2 recursive compile
             v
           C1234
```

Local level: four `TinyRes(64,8)`, 4096 learned parameters total.

Level 1: two recursively generated `TinyRes(64,16)`, 4096 parameters total.

Level 2: one `TinyRes(64,32)`, 4096 parameters.

Thus representation budget is exactly matched across hierarchy levels.

## Aggregate validation results

| condition | final NMSE vs original | geometric ratio vs direct | validation accuracy |
|---|---:|---:|---:|
| strict hierarchical frozen | 0.07138 | 1.8499x | 0.95556 |
| **hierarchical joint-adapt -> freeze -> recurse** | **0.04622** | **1.1992x** | **0.96543** |
| single-level recursive | 0.05037 | 1.3063x | 0.96420 |
| direct original -> single | 0.03853 | 1.0000x | 0.96296 |

## Main finding

The strict depth-2 route works numerically in all three seeds without returning to the original full-span target after Level 1, but error accumulation is visible: final/direct NMSE ratios were `1.8493x`, `1.9637x`, and `1.7433x`.

When the recursively generated `C12 -> C34` hierarchy is jointly adapted once against the original full span, then re-frozen and recursively compiled, the ratios fall to `1.1855x`, `1.2315x`, and `1.1812x`.

This joint-adapted depth-2 condition also beat the one-level recursive baseline in all 3 exploratory seeds.

## Interpretation

The result supports a specific mechanism hypothesis:

> Recursive depth is viable, but frozen recursive units accumulate approximation error. Re-opening the candidate boundary at the next hierarchy level, jointly adapting the recursively generated units to the wider span, and freezing again can substantially repair that accumulated error before the next Canaria-to-Canaria compilation.

This is closely aligned with the proposed freeze/unfreeze composition mechanism: the hierarchy is not treated as permanently immutable. Instead, each new composition boundary can become a temporary adaptation surface and is frozen again after alignment.

## Boundaries

C6 is exploratory only. It does not establish:
- a confirmatory depth-2 claim;
- unlimited recursive depth;
- lossless recursion;
- that the original teacher can be discarded indefinitely;
- architecture/task universality;
- LLM-scale behavior.

A fresh confirmatory depth-2 cohort is warranted.
