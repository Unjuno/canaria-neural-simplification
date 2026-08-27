# C2 freeze-boundary mapping — exploratory report

Status: **EXPLORATORY COMPLETE**. Do not promote into the public claim set without fresh confirmation.

## Question

C1 showed that jointly adapting an already-fitted three-Canaria cluster before freezing and recursively recompiling it into a matched single Canaria was much better than leaving the local candidates frozen. C2 asks whether that gain comes from a specific boundary (left, middle, right, or edges), or from giving the whole candidate cluster adaptation freedom.

Fresh exploratory seeds: `1320, 1321, 1322`. Held-out test data was not evaluated.

Candidate budget is exactly matched: three `TinyRes(64,8)` candidates = 3072 learned parameters; one `TinyRes(64,24)` = 3072 learned parameters.

## Aggregate results

| schedule | mean cluster NMSE vs original | mean recursive-single NMSE vs original | mean recursive val acc |
|---|---:|---:|---:|
| all_frozen | 0.16714 | 0.16689 | 0.94444 |
| left_only | 0.10562 | 0.10296 | 0.96173 |
| middle_only | 0.10704 | 0.10824 | 0.96049 |
| right_only | 0.10158 | 0.10286 | 0.95309 |
| left_middle | 0.06507 | 0.06993 | 0.96296 |
| middle_right | 0.06231 | 0.06821 | 0.96296 |
| edges_only | 0.06382 | 0.06813 | 0.96420 |
| **all_unfrozen** | **0.05177** | **0.06166** | 0.96049 |
| direct original -> single | — | **0.04625** | **0.96667** |

## Main exploratory observations

1. `all_unfrozen` had the lowest cluster NMSE in **3/3** seeds.
2. `all_unfrozen` also had the lowest recursive-single original-teacher NMSE in **3/3** seeds.
3. `edges_only` and `middle_right` were effectively tied among the two-candidate schedules on the aggregate recursive endpoint (`0.06813` vs `0.06821`). `left_middle` was somewhat worse (`0.06993`).
4. There was no stable left-versus-right one-candidate rule. `right_only` had the best mean cluster NMSE, but recursive ordering varied; `middle_only` was generally weak.
5. The strongest stable rule across C1 and C2 is therefore not "adapt the edges". It is: **jointly adapt the whole already-fitted candidate cluster against the wider span objective, then freeze it and recursively recompile from its input/output behavior.**
6. Recursive recompilation is still not lossless. `all_unfrozen` recursive mean NMSE was `0.06166`, versus `0.04625` for a matched single Canaria fitted directly to the original teacher. Mean excess = `0.01540`.

## Interpretation

The freeze schedule controls how well an existing collection of local Canaria candidates becomes a coherent intermediate representation. Increasing the adaptation freedom from zero -> one candidate -> two candidates -> all three produces a strong monotonic pattern in the aggregate, with full joint adaptation giving the best fidelity before and after recursive recompilation.

This supports a confirmatory hypothesis that a cluster of already-fitted Canaria candidates can serve as a reusable compiler IR after joint span-level adaptation, but with a measurable fidelity penalty relative to direct access to the original teacher.

## Boundary

C2 is exploratory mechanism evidence only. No test outcomes were inspected, and no universal hierarchical-compiler claim is supported yet. A fresh confirmatory cohort must lock the conditions and decision rules before outcomes.
