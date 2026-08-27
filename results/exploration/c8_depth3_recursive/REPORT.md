# C8 depth-3 recursive hierarchy and error scaling — exploratory report

Decision: **EXPLORATORY POSITIVE FOR RE-ALIGNMENT; STRICT DEPTH-3 DEGRADES STRONGLY**.

C8 extends the recursive tree to eight original residual blocks while holding the learned replacement budget fixed at 4096 parameters at every hierarchy level.

Fresh seeds: `1390–1392`. No held-out test evaluation.

## Constant-budget tree

```text
8 x TinyRes(64,4)   = 4096 params
        ↓ pair align + recursive compile
4 x TinyRes(64,8)   = 4096 params
        ↓
2 x TinyRes(64,16)  = 4096 params
        ↓
1 x TinyRes(64,32)  = 4096 params
```

Two paths share the same Level-1 recursive units:

- **strict**: after Level 1, never return to the original 4-block or 8-block span targets;
- **realign-each-level**: temporarily unfreeze each newly formed hierarchy boundary, jointly align it to the corresponding wider original span, freeze, then recursively compile from hierarchy outputs only.

## Aggregate results

| condition | mean final val NMSE | geometric ratio vs direct | mean val accuracy |
|---|---:|---:|---:|
| strict depth-3 | 0.21937 | 4.5762x | 0.92593 |
| **realign each level** | **0.05627** | **1.1740x** | **0.96667** |
| single-level recursive | 0.06917 | 1.4431x | 0.96173 |
| direct original -> single | 0.04793 | 1.0000x | 0.96667 |

The realigned depth-3 route beat the single-level recursive control in all 3 exploratory seeds.

## Depth trajectory

Mean full-span NMSE:

| hierarchy state | strict | re-align path |
|---|---:|---:|
| Level-1 recursive units chained | 0.22095 | 0.22095 |
| Level-2 units chained | 0.21826 | **0.08674** |
| Level-3 final | 0.21937 | **0.05627** |

The strict path's large error appears immediately when independently generated Level-1 pair units are chained out of their original input distributions. Further strict recursive recompilation largely preserves this error instead of repairing it.

By contrast, re-aligning the newly formed 4-block boundaries before Level-2 compilation cuts mean NMSE from about 0.221 to 0.087. Re-aligning the final 8-block boundary before Level-3 compilation reduces it further to about 0.056.

## Per-seed final/direct ratios

Strict depth-3:
- 1390: **4.4632x**
- 1391: **4.5961x**
- 1392: **4.6717x**

Re-align each level:
- 1390: **1.1847x**
- 1391: **1.2297x**
- 1392: **1.1106x**

## Interpretation

The main mechanism suggested by C8 is not simply “more recursive levels cause more approximation error.” The dominant failure mode in the strict tree is **boundary/distribution mismatch**: a recursively generated unit is trained on the original activation distribution at its own span entrance, but later receives outputs produced by another approximate recursive unit.

The proposed freeze/unfreeze composition method addresses exactly this failure mode:

```text
compose children
      ↓
temporarily unfreeze the new boundary
      ↓
jointly align to the wider original span
      ↓
freeze again
      ↓
recursive Canaria-to-Canaria compile
```

In these exploratory seeds, repeating this operation at every newly formed hierarchy level keeps depth-3 fidelity close to a direct matched replacement and outperforms one-shot full-span adaptation of all eight local candidates.

## Boundaries

C8 is exploratory. It does not establish:
- a confirmatory depth-3 claim;
- a universal depth-error scaling law;
- unlimited recursion;
- losslessness;
- architecture/task universality;
- LLM-scale behavior.

A fresh depth-3 confirmatory cohort is warranted before promoting this mechanism beyond exploratory status.
