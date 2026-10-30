> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Joint-control budget-grid results — 2026-10-27

## Finding
On the sixteen reused seen-digits teachers, single-map Comp (C) reached the locked quality threshold at a smaller first-passing candidate-grid budget than jointly fitted two-map J in 11 cases, tied in 4, and required more in 1. The geometric C/J ratio is **0.788324**, paired bootstrap 95% interval **[0.694450, 0.904368]**. Mean log2 ratio is −0.343139; interval [−0.526057, −0.145019]. The predeclared outcome is `CONDITIONAL_GRID_ADVANTAGE`.

This is an adaptive exploratory extension conditional on already seen digits, validation data and teachers, not fresh-data confirmation, objective-only causality, global minimum capacity, or a deployment benchmark. The old Comp/local ratio **0.5316** used a different cohort/recipe and comparator. Its numerical difference from **0.7883** cannot be attributed solely to replacing the weak local control with joint fitting.

## Frozen design and execution
Protocol/seeds frozen at 5d5bbd4; reviewed runtime **4c9e277d5f1ee913a8a75f1eadeb5c6f0ab9e96a**. Parent teacher runtime **6d9e60a127aa83f3e966ced17767a980441593db**; predecessor results 824aca4568132f6b68e27ae2e5d860910af8f4a7. All sixteen original teachers, targets and schedules were reused without retraining. The 32 original 2048-budget J/C outcomes were copied verbatim, not counted as new independent replications. Exactly **192 new fits** completed: 16 teachers × 6 missing budgets × 2 arms. The full grid has **224 cells**.

Seven fixed budgets: 512, 1024, 1536, 2048, 3072, 4096, 6144 weights; J two maps of width h, C one map of width 2h. Each new fit used 600 updates, identical parent sample schedule and fixed AdamW recipe. All widths were run before validation; no adaptive width choice, early success, retry, checkpoint selection or test-set evaluation. Quality requires NMSE ≤ 0.08 AND accuracy ≥ same-teacher accuracy − 0.02. Every teacher remains in the cohort.

All sixteen subprocesses exited 0, no aborted or not-started members, no scientific retries. Both arms passed at some budget for all 16, so the full-cohort ratio gate opened; C-only/J-only/neither counts are all zero. Bootstrap: 100,000 paired-teacher resamples, seed 2710276144. Equal weights and updates are not equal compute.

The one cohort finished 2026-10-27 01:11:55.793131 UTC. Peak recorded worker RSS 415,100 KiB, stored raw evidence 25,509,645 bytes; observed teacher-body durations 5.49–6.04 s. These are run-resource observations, not deployment speed measurements. Caps remained two concurrent workers, 4 GiB each, 1 GiB evidence and three hours aggregate within the existing common deadline.

## Per-budget quality passes (out of 16)
| Budget | C | J | Provenance |
|---:|---:|---:|---|
| 512 | 0 | 0 | New |
| 1024 | 0 | 0 | New |
| 1536 | 10 | 2 | New |
| 2048 | 15 | 12 | Reused |
| 3072 | 14 | 14 | New |
| 4096 | 16 | 15 | New |
| 6144 | 16 | 16 | New |

## First passing observed candidate budget
| Teacher ID | C | J |
|---:|---:|---:|
| 0 | 2048 | 3072 |
| 1 | 3072 | 2048 |
| 2 | 1536 | 1536 |
| 3 | 1536 | 2048 |
| 4 | 1536 | 2048 |
| 5 | 2048 | 3072 |
| 6 | 1536 | 2048 |
| 7 | 1536 | 2048 |
| 8 | 2048 | 2048 |
| 9 | 1536 | 2048 |
| 10 | 1536 | 1536 |
| 11 | 1536 | 2048 |
| 12 | 2048 | 4096 |
| 13 | 2048 | 2048 |
| 14 | 1536 | 3072 |
| 15 | 1536 | 2048 |

## Nonmonotonic outcomes retained
- Teacher 0, C: 2048 passes, 3072 fails, 4096 and 6144 pass.
- Teacher 4, C: 1536/2048 pass, 3072 fails, 4096 and 6144 pass.
- Teacher 1, J: 2048 passes, 3072/4096 fail, 6144 passes.

No interpolation or monotonic correction was applied. A first-passing tested width is not a mathematical minimum. `CELLS.csv` contains every one of the 224 scalar outcomes and explicit reuse flags, including failures.

## Evidence and verification status
The author independently recomputed saved-cell metrics, quality decisions and the fixed bootstrap without fitting models again: 3,235 checks passed, covering 882 raw artifact hashes and parent integrity. This is **author verification**, not non-author acceptance. The separately assigned non-author audit subsequently passed **5,823 checks with zero discrepancies**, decision **PASS_WITH_STATED_LIMITS**. Its complete report, script and manifest are preserved under `independent_audit/`. It recomputed saved-array metrics and the bootstrap without model execution; source/log/hash consistency is not a replay of every trajectory or continuous concurrency measurement.

Raw SUMMARY SHA-256: `b38d6f2175915cda5f738504299f8e7421b0579c6b0eb652fe23109d4e51d1ca`.

## Remaining scientific limits
One initialization per new width/teacher, shared validation data, adaptive question selection after predecessor results, a sparse finite grid and potential architecture/optimization differences limit inference. C versus J differs in structure as well as training parameterization. Generalization to unseen datasets, independently chosen teachers, alternative recipes or practical runtime remains untested. No additional fit is authorized by this report.
