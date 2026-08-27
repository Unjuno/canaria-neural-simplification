# C4 — SmallViT cross-family recursive Canaria composition

Status: **EXPLORATORY COMPLETE**. No held-out test evaluation.

Fresh exploratory seeds: `1340–1342`; all 3 met the locked SmallViT teacher eligibility rule (`validation accuracy >= 0.95`). Cluster and single replacement budgets were exactly matched at **4096 learned parameters**.

## Main result

Mean validation results:

| schedule | cluster NMSE vs original | recursive single NMSE vs original | recursive val acc |
|---|---:|---:|---:|
| all frozen | 0.25046 | 0.25782 | 0.77284 |
| left only | 0.22430 | 0.23869 | 0.81481 |
| right only | 0.22290 | 0.23816 | 0.79630 |
| **all unfrozen** | **0.20516** | **0.22920** | 0.82469 |
| direct original single | — | **0.21745** | **0.82593** |

`all_unfrozen` had the lowest recursive-single original-teacher NMSE in **3/3** seeds. The recursive/direct NMSE ratios were `1.0373`, `1.0605`, and `1.0580`, geometric mean **1.0519x**.

Thus the qualitative C1–C3 mechanism transferred to a Transformer-family teacher: jointly adapting the local candidate cluster before freezing it made later cluster-only recursive recompilation consistently better than recompiling the unadapted cluster.

## Important negative/boundary result

This exact-matched token-wise residual grammar is substantially weaker than the SmallViT teacher in downstream utility. Mean teacher validation accuracy was `0.95679`, while all-unfrozen recursive and direct-original single replacements were `0.82469` and `0.82593` respectively.

C4 therefore supports a **cross-family mechanism signal**, not a high-utility Transformer replacement claim. A fresh confirmatory cohort should compare recursive versus direct/frozen conditions under this fixed grammar; improving the grammar is a separate future experiment and must not be tuned on C4 confirmatory seeds.
