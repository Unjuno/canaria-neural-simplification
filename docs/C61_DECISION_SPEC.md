# C61 imported decision specification

Status: **decision specification reconstructed from the 2026-09-02 handoff; not the original preregistration artifact**.

This file translates the imported C61 conditions into an explicit H/T/D/C/U record without pretending that its GitHub timestamp predates fresh outcomes.

## H — falsifiable hypothesis

Within the imported Residual-CNN Gaussian `sigma=.04` testbed, a P4/32 teacher-correction interface is non-inferior to P8/32 under both prospectively reported gates:

1. validation-accuracy P4−P8 paired bootstrap95 lower bound exceeds `-2 percentage points`;
2. P4/P8 validation-NMSE geometric-mean ratio paired bootstrap95 upper bound is below `1.25`.

The hypothesis is scoped to the exact architecture, boundary, shift, calibration, basis, subset, replacement grammar, optimization, and eligibility rules of the original C61 runner.

## T — minimum valid test

Imported fixed conditions:

- attempted fresh model seeds: `49400–49415`;
- minimum eligible models: `8`;
- P4 versus P8;
- `192` calibration samples;
- Gaussian shift `sigma=.04`;
- same nested QR basis and same calibration subset across dimensions;
- `100000` paired bootstrap resamples;
- held-out test unused.

All attempted seeds must be retained in the audit record. No rescue seeds, changed margins, alternate basis, alternate calibration count, or post-outcome protocol substitution is allowed.

A valid scientific test additionally requires recovery of the original C61 runner/protocol provenance and raw fresh rows. A new runner can be a replication/reconstruction experiment, but must not be mislabeled as the original C61.

## D — decision rule

- `STOP_INSUFFICIENT_ELIGIBLE` if eligible count `< 8`.
- `C61_CONFIRMATORY_PASS` only if both locked gates pass.
- `C61_CONFIRMATORY_FAIL` if eligibility is sufficient and either locked gate fails.
- Repository status remains **UNRESOLVED/PROVENANCE-INCOMPLETE** if raw rows, original runner provenance, or the exact bootstrap RNG/implementation needed to reproduce the original decision cannot be recovered.

The last state is a provenance state, not a statistical rescue category.

## C — principal counter-hypotheses / failure modes

If C61 fails, plausible competing explanations include:

- P4 genuinely removes task-relevant correction directions that P8 retains;
- C60 exploratory advancement was an optimistic cohort realization;
- P4 is more sensitive to the fixed basis/subset geometry;
- optimization noise or conditioning grows sharply at P4;
- the Gaussian shift excites a boundary subspace whose effective rank is above four;
- the relevant architecture effect is actually a boundary/grammar/calibration interaction rather than architecture itself.

A failure would not invalidate C59 P8 non-inferiority.

## U — uncertainty and provenance

Known unresolved sources:

- original C59/C60/C61 runner files are absent from the current GitHub evidence surface;
- C59/C60 raw seed rows and independent-audit artifacts have not yet been imported;
- C61 fresh seed rows are absent;
- the handoff specifies `100000` bootstrap resamples but not the exact bootstrap RNG seed / implementation detail;
- the precise execution environment and hardware for C59–C61 have not yet been recovered into this branch;
- GitHub Actions history searched for 2026-09-01 through 2026-09-03 did not expose the original experiment execution.

Because these are provenance/systematic uncertainties, they cannot be represented honestly by a numerical combined standard uncertainty `u_c` or coverage factor `k` from the currently available information. Assigning such numbers would fabricate metrological information. The correct present state is qualitative provenance uncertainty plus the reported bootstrap intervals from the imported handoff.

## Next gate

Do not start a P2/P1 follow-up under the C59–C61 lineage until C61 is resolved or explicitly declared unrecoverable. If unrecoverable, a separately named fresh replication may be locked using recovered/reimplemented code, with its own seeds and protocol chronology.
