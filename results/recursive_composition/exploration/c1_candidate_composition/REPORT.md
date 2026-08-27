# C1 — freeze-schedule composition of adjacent Canaria candidates

Status: **EXPLORATORY COMPLETE**. Do not use as confirmatory/public evidence.

## Question

Can three already-fitted adjacent Canaria candidates be jointly re-adapted under a frozen surrounding network and then collapsed into one matched-budget Canaria using only the candidate cluster as the new teacher?

## Design

- residual-MLP digits testbed;
- studied span: first 3 residual blocks;
- three local `TinyRes(64,8)` candidates = 3072 learned parameters total;
- one `TinyRes(64,24)` composed replacement = 3072 learned parameters;
- seeds 1310–1312;
- no held-out test evaluation;
- Stage 2 varies only which candidate modules are unfrozen for span-target adaptation;
- Stage 3 freezes the resulting cluster and fits a fresh single Canaria to `a0 -> cluster(a0)`, without using original teacher `a3` as the Stage-3 target.

## Mean exploratory results

| Stage-2 freeze schedule | cluster NMSE vs original teacher | recompiled single NMSE vs cluster | recompiled single NMSE vs original teacher | recompiled val acc |
|---|---:|---:|---:|---:|
| all frozen | 0.15533 | 0.00971 | 0.15627 | 0.94815 |
| middle only unfrozen | 0.09847 | 0.01449 | 0.10078 | 0.95679 |
| edges only unfrozen | 0.05771 | 0.02143 | 0.06303 | 0.96296 |
| all three unfrozen | **0.04629** | 0.02520 | **0.05755** | **0.96420** |
| direct original-teacher single | — | — | **0.04405** | **0.96914** |

For both cluster NMSE and recompiled-single NMSE versus the original teacher, all three exploratory seeds had the identical ordering:

`all_unfrozen < edges_only < middle_only < all_frozen`.

## Interpretation

The result supports the working hypothesis that **freeze scheduling materially controls whether a set of adjacent Canaria candidates forms a useful composable intermediate representation**.

The strongest exploratory schedule was to jointly unfreeze all three candidates while the surrounding network stayed fixed, then re-freeze the cluster and train a single matched-budget Canaria using only cluster outputs. This recursive Canaria-to-Canaria path retained most of the fidelity gain and nearly matched the direct-control validation utility.

The result is not lossless. Direct fitting to the original teacher achieved mean NMSE 0.04405 versus 0.05755 for the best recursive path. Thus C1 does not establish that an arbitrary candidate cluster can replace the original teacher as a perfect compiler IR.

A second pattern is also important: the untouched local cluster was easiest to imitate *relative to itself* (lowest `single vs cluster` NMSE), but it was far from the original teacher. As cluster adaptation improved original-teacher fidelity, the cluster became somewhat harder for the fixed single grammar to imitate. This suggests a **fidelity-versus-recompilability tradeoff** rather than a monotonic notion of 'easier composition'.

`edges_only` consistently recovered much more than `middle_only`, motivating a later boundary-localization experiment (left edge vs right edge vs adjacent pairs) if this line is continued.

## Evidence boundary

- exploratory seeds only;
- one dataset/model/span/grammar;
- no test-set evaluation;
- no public hierarchical-compiler claim;
- fresh protocol and fresh seeds are required before promotion.

Per-seed values are in `seed_rows_corrected.csv`. The original `seed_rows.csv` contains a documented manual transcription error in the `cluster_val_acc` column only and is retained for provenance; see `SEED_ROWS_CORRECTION.md`.
