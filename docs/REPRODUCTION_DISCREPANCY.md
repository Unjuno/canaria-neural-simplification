# Core cohort reproduction discrepancy — publication blocker

**Status: unresolved strict numerical reproduction mismatch.** Tracking: [Issue87](https://github.com/Unjuno/canaria-neural-simplification/issues/87), parent Issue13. The previously successful seed1200 smoke test did not establish eight-seed reproduction.

The scientific runner, original protocol and archived summary are unchanged. We reran all already-observed seeds1200–1207. This is software reproduction, not new confirmatory evidence.

| Source / execution | Mean component-wise budget | Mean composed budget | Geometric paired budget ratio | Composed smaller |
|---|---:|---:|---:|---:|
| Historical saved summary | 3584 | 1728 | 0.482339 | 8/8 |
| GitHub pinned modern-stack repeat | 3456 | 1728 | 0.510732 | 8/8 |
| Local previous-version-stack repeat | 3072 | 1792 | 0.577350 | 8/8 |

Both repeats satisfy the original **directional** budget and test-utility gate pattern, but selected endpoints/test differences do not strictly match the archived values. Neither repeat is labeled exact reproduction PASS. These differences exceed rounding tolerance and are not resolved by widening tolerances.

## Execution conditions

GitHub run34173091549, source6eaee565f00c16efb805d39c037ad025294f1ef1: Ubuntu24.04-family runner / AMD EPYC7763, Python3.11.16, torch2.13.0+cpu, NumPy2.4.6, scikit-learn1.9.0, one Torch thread. Local repeat: AMD EPYC9V74, Python3.13.5, torch2.10.0+cpu, NumPy2.3.5, scikit-learn1.8.0, one thread per process (two independent processes concurrently). Clocks are not fixed; no timing/resource benchmark is claimed. All original model/fit budgets, seeds and decision thresholds remain unchanged.

A second same-local-stack execution of known seeds1202 and1205 produced byte-identical complete result JSONs. This is limited repeatability evidence, not proof of cross-platform parity or the cause of the original discrepancy.

## What is unknown

Versions, CPU instruction dispatch, numerical backend and original execution provenance are not fully separated. The original full per-seed output archive and exact source/environment should be recovered where possible. We do not claim that the old data are fabricated, that a specific library caused the mismatch, or that the directional composition effect is invalidated.

The runner's **candidate budget selection excludes test metrics**. It evaluates selected candidate endpoints, a prespecified mechanistic control, and the teacher on test. Therefore “no test is accessed anywhere before endpoint selection” is too strong a description; that wording is corrected without changing the runner.

## Preserved evidence and next step

[Reports and raw records](../results/reproduction/publication_2026-09-08/README.md) include the initial FAIL report, both environments, all local raw outputs and repeatability checks. The enhanced verifier now retains every fresh-process output/log and checks first-passing endpoint selection and missing/duplicate/incomplete cohorts. Its old numerical tolerances are not loosened.

The strict announcement gate remains **BLOCKED**. Code inspection and a clearly labeled research preview can be useful, but do not advertise the exact historical magnitudes as fully reproduced under the modern pinned stack. Repository cleanup does not close Issue87 or Issue13.
