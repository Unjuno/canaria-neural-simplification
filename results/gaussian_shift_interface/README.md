# Gaussian-shift interface evidence

This directory is a **research-only import/recovery surface** for C59–C61. It is not part of the reviewed public claim registry on `main`.

## Evidence classes

- `IMPORTED_HANDOFF_RESULT` — aggregate outcome transcribed from a prior execution session. It is not independently reconstructed here.
- `IMPORTED_LOCKED_PROTOCOL` — protocol reported as locked before fresh outcomes, but imported to GitHub after that lock. The GitHub commit time is not a preregistration time.
- `GITHUB_RECONSTRUCTED` — may be used only after raw rows/artifacts are present and an independent script reconstructs the reported statistics from those rows.

At initial import, C59/C60 are `IMPORTED_HANDOFF_RESULT`; C61 is `IMPORTED_LOCKED_PROTOCOL`; none is `GITHUB_RECONSTRUCTED`.

## Layout

- `HANDOFF_C59_C60.json` — machine-readable imported aggregate outcomes.
- `c61/IMPORTED_PROTOCOL.json` — C61 imported locked conditions and explicit missing-artifact list.
- `c61/SEED_ROWS_TEMPLATE.json` — schema/example only; **not evidence** and contains no fabricated outcome values.

## C61 intake rule

A fresh result row may enter an evidence file only if it can be tied to the original C61 runner/execution provenance. Do not manually fill plausible values and do not substitute a newly reconstructed runner while calling it the original C61.

Expected cohort: seeds `49400–49415`. All attempted seeds must be retained. Eligibility is recorded independently of P4/P8 outcomes. Formal evaluation stops if fewer than 8 are eligible.

The imported locked decision gates are:

- paired validation accuracy P4−P8: bootstrap 95% CI lower bound must exceed `-2 percentage points`;
- P4/P8 NMSE geometric-mean ratio: bootstrap 95% CI upper bound must be below `1.25`;
- `100000` bootstrap resamples;
- held-out test is not used.

The exact bootstrap RNG seed/implementation is still a provenance gap and must be recovered rather than invented.

## Promotion rule

Do not convert imported handoff values into reviewed/public evidence merely by copying them into this directory. Promotion requires raw-artifact recovery, independent reconstruction, C61 resolution, and separate scientific review.
