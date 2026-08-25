# Training-time consolidation results index

This directory contains the compact machine-readable public indexes for the Canaria training-time consolidation program.

## Files

- `summary.json` — G7–G17 headline outcomes.
- `protocol_manifest.json` — G7–G17 fresh seed ranges, evidence classes, decision rules, and available protocol SHA256 values.
- `late_stage_summary.json` — G18–G26 headline outcomes and recorded protocol/result SHA256 identifiers. G27 remains exploratory and is documented in `docs/LATE_STAGE_FINDINGS.md`.
- `ARTIFACT_INVENTORY.md` — explains which evidence is present directly versus indexed only by retained artifact hashes, including known retention limitations.

## Earlier confirmatory sequence: G7–G17

| experiment | question | fresh seeds | decision |
|---|---|---|---|
| G7 | progressive consolidation vs small-from-start / one-shot | 4300–4307 | PASS |
| G8 | does correct function-aligned transfer matter? | 4500–4507 | PASS |
| G9 | how much transfer fit is useful? | 4700–4707 | PASS |
| G10 | can structured weight inheritance replace functional fitting? | 4900–4907 | PASS: inheritance alone insufficient; hybrid best |
| G11 | can a calibration-only controller autonomously reach the target architecture? | 5400–5407 | PASS |
| G15 | staged `4→3→2` vs waiting for direct `4→2` | 5800–5807 | PASS |
| G17 | does fit factorization alone reproduce staged benefit? | 6000–6007 | PASS equivalence: factorization alone does not reproduce benefit |

The strongest mechanism separation in this section is G15 + G17: task learning/recontracting between consolidation events matters; merely splitting one compiler fit into two does not reproduce the staged advantage.

## Late-stage confirmatory / boundary sequence: G18–G26

| experiment | question | fresh seeds | decision |
|---|---|---|---|
| G18 | does remaining learning horizon improve autonomous commit timing? | 6200–6211 | PASS |
| G19 | is the staged path effect specific to `4→3→2`? | 6600–6607 | PASS on `5→4→2` vs `5→2` |
| G20d | does recontracting reduce next-compiler optimization cost at matched relative error? | 7300–7307 | PASS |
| G20e | does matched relative compiler error imply matched task safety? | 7500–7507 | CONFIRMED BOUNDARY: no |
| G21 | does a hard shadow-damage veto improve the controller? | 7800–7811 | FAIL |
| G22 | does recontracting increase downstream sensitivity while fitting becomes easier? | 8100–8111 | PASS |
| G23 | does gradient/error direction improve immediate task-damage prediction? | 8300–8311 | PASS |
| G24 | does a logit-space second-order term improve prediction further? | 8400–8411 | PASS |
| G25 | does the fixed G24 predictor transfer to a different depth path without refitting? | 8500–8507 | PASS |
| G26 | does remaining horizon improve future-damage prediction? | 8700–8711 | PASS |

G27 tested fixed future-risk caps as a compiler-budget controller. It remains exploratory: strict caps spent more compiler work for better utility, while loose caps saved compiler work but worsened utility. No cost/utility Pareto improvement was established.

## Current interpretation

The combined training-time evidence supports a two-sided recontracting picture in the tested small real-text LM:

1. after consolidation and intermediate task learning, the next compiler can become easier to optimize in normalized functional-error terms;
2. the downstream task can simultaneously become more sensitive to the residual approximation error;
3. therefore commit risk depends on error direction/sensitivity and remaining learning horizon, not only on one scalar NMSE threshold.

This does **not** establish universality to large pretrained models or exact FLOP/energy/runtime gains.

## Evidence integrity

Do not infer that every protocol/result artifact is stored as a separate file here merely because a SHA256 is recorded. Read `ARTIFACT_INVENTORY.md` before making protocol-integrity claims.

For project-level interpretation, start with:

- `../../docs/CORE_DISCOVERY.md`
- `../../docs/TRAINING_TIME_CONSOLIDATION.md`
- `../../docs/LATE_STAGE_FINDINGS.md`
- `../../docs/NEGATIVE_RESULTS.md`
