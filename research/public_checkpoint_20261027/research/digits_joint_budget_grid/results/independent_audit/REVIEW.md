> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Independent saved-output audit — Canaria joint budget grid

Decision: **PASS WITH STATED LIMITS**. Final run: **5,823 checks, zero discrepancies**.

Runtime: `4c9e277d5f1ee913a8a75f1eadeb5c6f0ab9e96a`.
Raw SUMMARY SHA256: `b38d6f2175915cda5f738504299f8e7421b0579c6b0eb652fe23109d4e51d1ca`.

## Verified from saved evidence

- 192 new fits plus 32 verbatim reused 2048-budget cells: C and J, 16 teachers, seven budgets (512, 1024, 1536, 2048, 3072, 4096, 6144).
- Independently recalculated all validation NMSE, MSE, denominator, SSE, centered SS, argmax correct counts, accuracy, and quality flags from saved arrays. Quality combines NMSE <= .08 and accuracy >= teacher accuracy - .02; counts use 270 validation items.
- All 16 teachers have a passing C and J budget, so no censoring prevents the complete-cohort ratio. Three arm/teacher pass sequences are nonmonotonic; minima use the actual grid, not interpolation.
- First-passing C/J geometric budget ratio: **0.7883241833589479**, paired 95% bootstrap CI **[0.694449973999738, 0.904367624088204]**. Independently recomputed 100,000 paired draws using declared numpy RNG seed 2710276144. Mean log2 ratio -0.34313906222956647; CI [-0.5260573241680421, -0.14501874963942193]. C smaller in 11, tied in 4, larger in 1.
- New-fit saved states have the declared parameter counts. Recorded schedules are 600 x 128; each fit has 600 sequential finite update-loss records and recorded parameter-step counts of 600. Initial/final numeric state hashes match; final loss agrees with the last log entry.
- Environment records match Python 3.13.5, torch 2.10.0+cpu, numpy 2.3.5, scipy 1.17.0, sklearn 1.8.0, deterministic single-thread AVX2 settings. Recorded time/RSS, aggregate recorded time, saved size, and completion timestamp are within the declared caps/deadline.

## Scope and limits



This is an adaptive extension using the same seen digits/teachers and 32 reused cells, not fresh-data confirmation. The confidence interval is conditional on this cohort and fixed procedure. Grid minima are not global capacity minima. No objective-only causal attribution, deployment claim, or external-generalization conclusion follows. The earlier C/local ratio has a different denominator and must not be substituted for the present C/joint ratio.

Artifacts: `audit.py` is the independent executable check; `AUDIT.json` contains results and per-teacher minima; `MANIFEST.json` binds this review and the audit artifacts. No external publication or user notification was performed by this reviewer.
