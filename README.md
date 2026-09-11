# Canaria

**Task-conditioned neural simplification — research code, evidence, and limitations.**

[日本語](README.ja.md) · [Evidence](docs/CLAIMS_AND_EVIDENCE.md) · [Readiness](docs/ANNOUNCEMENT_READINESS.md) · [Research index](docs/RESEARCH_INDEX.md)

Canaria studies whether a trained neural-network span can sometimes be replaced by a smaller learned computation when the whole span is fitted as one input–output function instead of simplifying its implementation blocks independently.

**This is a bounded research preview, not a universal compression method or a production inference library.** The current reviewed headline is a fresh direct baseline under one residual-MLP / sklearn-digits / first-two-block protocol. A separate California Housing experiment provides limited task/dataset external-validity support within the same architecture family.

## Current reviewed direct baseline: R87R2

Under the prospectively fixed `R87R2_FRESH_SQRT_PROFILE_CONFIRMATION` protocol and the tested `portable_v2_sqrt64` CPU numerical profile, all 16 fresh model seeds (`871200`–`871215`) were eligible.

- geometric composed/component-wise selected replacement-budget ratio: **0.5316**
- paired 95% bootstrap CI for mean `log2(B_composed/B_componentwise)`: **[-1.0413, -0.7500]**
- composed selected a strictly smaller passing budget in **15/16** seeds, with one tie
- paired 95% bootstrap CI for selected test-accuracy difference, composed minus component-wise: **[+0.00028, +0.00750]**
- locked decision: **`R87R2_CONFIRMATORY_PASS`**

“Selected budget” is the first passing point on the locked finite grid, not a mathematical minimum. The 16 observations are model seeds on one reused digits split, not 16 independent datasets. Test metrics did not select the replacement budgets, but the test split is reused rather than an operationally unseen external dataset. The numerical profile is a scoped tested CPU research recipe, not a universal CPU/GPU portability theorem.

[Protocol](results/reproduction/r87r2_fresh_sqrt_profile/PROTOCOL.json) · [Fresh raw rows](results/reproduction/r87r2_fresh_sqrt_profile/FRESH_ROWS.json) · [Decision](results/reproduction/r87r2_fresh_sqrt_profile/DECISION.json) · [Candidate audit](tools/audit_publication_candidate.py)

## Historical 1200–1207 archive: retained, not rewritten

The older recorded digits cohort showed the same directional pattern, including 3,584 versus 1,728 mean selected replacement parameters and composed lower in 8/8 recorded seeds. However, modern full-cohort reruns did **not** reproduce all archived per-seed endpoints/statistics exactly. See the [historical reproduction discrepancy](docs/REPRODUCTION_DISCREPANCY.md).

R87R2 is a **new confirmatory cohort under a new scoped numerical profile**. It is not a recovery of the historical 1200–1207 values, and Issue #87 remains open as historical reproduction/provenance debt. The archived values are preserved rather than silently replaced.

## Bounded regression support: California Housing Phase4

On one fixed California Housing split, using the same residual-MLP family and a teacher recipe selected prospectively in Stage A without using the held-out test split, the fresh eight-seed Stage-B cohort produced `PHASE4_CONFIRMATORY_PASS`.

- composed selected a strictly smaller passing replacement budget in **8/8** seeds
- geometric composed/component-wise selected-budget ratio: **0.4760**
- paired 95% bootstrap CI for mean log2 budget ratio: **[-1.3282, -0.8325]**
- paired 95% bootstrap CI for selected held-out-test R² difference, composed minus component-wise: **[+0.00436, +0.00967]**
- teacher held-out-test R² mean 95% CI: **[0.7911, 0.7996]**
- a second hosted worker reproduced the eight scientific records exactly

This is one dataset and one split. It supports limited task/dataset external validity, not architecture universality. Selected budgets remain finite-grid minima.

[Stage-B protocol](results/phase4/california_regression/STAGE_B_CONFIRMATORY_PROTOCOL.json) · [Fresh raw rows](results/phase4/california_regression/stage_b_confirmatory/FRESH_ROWS.json) · [Decision](results/phase4/california_regression/stage_b_confirmatory/DECISION.json)

## Reproduce and audit the evidence classes separately

For the reviewed publication candidate, install NumPy 2.3.5 and run:

```bash
python tools/audit_publication.py
python tools/audit_publication_candidate.py
```

The second command recomputes the R87R2 and Phase4 locked aggregate decisions from the vendored raw rows and verifies that the vendored evidence retains the exact Git blob identities reviewed on the source research commits.

The historical strict 1200–1207 reproduction check remains available separately:

```bash
python tools/audit_publication.py --require-reproduction
```

That path is expected to remain blocked while Issue #87 is unresolved. It is deliberately **not** redefined as a PASS by R87R2. [QUICKSTART.md](QUICKSTART.md) documents the historical core rerun path.

## What is — and is not — supported

The reviewed evidence supports a bounded empirical composition-budget effect for declared replacement spans under specific protocols. It does **not** establish a universal complexity law, whole-model compression, LLM-scale applicability, a universal minimum interface dimension, runtime speedup, RAM/VRAM reduction, energy reduction, or hardware-wide portability.

[SmallViT direct-composition evidence](docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md) remains supporting architecture-family evidence under its own narrow two-block/digits boundary. Systems S1–S7 remain prototype measurements only. Imported C59/C60 are Residual-CNN evidence; original C61 remains unresolved and is not repaired by the differently architected Residual-MLP C61R line.

Negative and invalidated evidence stays visible. Phase3B is excluded from a positive stronger-teacher headline because its preregistered teacher-strength gates were uncertain. Phase3C is retained as boundary evidence. Phase2E remains **`INVALIDATED_IMPLEMENTATION_BUG` / `DO_NOT_USE_FOR_INFERENCE`**; Phase2I causal attribution remains retracted; Phase2O did not establish a repair-sample advantage. [Negative results](docs/NEGATIVE_RESULTS.md)

## Publication state

The machine-readable selection is in [publication/CLAIM_POLICY.json](publication/CLAIM_POLICY.json), with the independent post-v0.2 disposition ledger in [publication/POST_V02_CLAIM_LEDGER.json](publication/POST_V02_CLAIM_LEDGER.json). Merge, release tagging, announcement, Issue #13 closure, external reproduction, and peer review are separate events; a CI PASS does not claim any of them.

See [STATUS.md](STATUS.md) and [announcement readiness](docs/ANNOUNCEMENT_READINESS.md) for the current gate. The frozen tag `v0.2.0-public-snapshot` remains historical provenance, not a certificate of current readiness.

## Contribute and cite

See [CONTRIBUTING.md](CONTRIBUTING.md). Reproduction failures and well-controlled counterexamples are useful contributions. Cite the **exact commit and experiment artifacts**, not an unversioned claim. [CITATION.cff](CITATION.cff)

Original code and documentation: [Apache-2.0](LICENSE). Third-party data and dependencies retain their own licenses.
