# C61R — prospective P4 vs P8 Gaussian-shift replication

Status: **C61R_CONFIRMATORY_PASS**

Evidence class: `GITHUB_PROSPECTIVE_CONFIRMATORY`.

This is a new prospective replication. It is **not** the missing original C61 execution and must not be cited as reconstructing that experiment.

## Locked design

The protocol was committed before fresh outcomes at `1aad40d986ae5bf99bee72d7f57d03b23ce0b849`.

- fresh seeds: `59400–59415`;
- eligible: `16/16`;
- additive Gaussian input shift: `sigma=0.04`;
- calibration samples: `192`;
- comparison: nested-QR P4 versus P8 using the same basis and shifted tensors within seed;
- validation non-inferiority margin: `-2 percentage points`;
- P4/P8 NMSE-ratio margin: `1.25`;
- paired percentile bootstrap: `100000` resamples, RNG seed `3960519679`;
- held-out test: not used.

## Primary result

Validation accuracy, P4 minus P8:

- mean: `-0.3472228 pp`;
- bootstrap95: `[-0.6250005, -0.0925928] pp`;
- locked margin: `-2.0 pp`;
- decision: **PASS**.

The interval is entirely below zero, so this result does **not** support equality or superiority of P4. It supports the narrower claim that the observed P4 validation-accuracy loss remained within the prospectively fixed non-inferiority margin.

P4/P8 NMSE ratio:

- geometric mean: `1.0135635`;
- bootstrap95: `[0.9993515, 1.0291186]`;
- locked margin: `1.25`;
- decision: **PASS**.

Informative P4 NMSE change versus the frozen compiled hierarchy averaged `-0.00173790`; this was not a primary locked gate.

Both primary gates passed, with `16/16` eligible seeds and no missing fresh rows. Therefore the locked overall decision is **C61R_CONFIRMATORY_PASS**.

## Scope

Supported wording:

> Under the exact C61R Residual-MLP recursive-hierarchy protocol on sklearn digits with additive Gaussian input noise sigma=0.04, a nested-QR P4 correction interface was non-inferior to P8 under the prospectively fixed validation-accuracy and NMSE-ratio margins.

Unsupported wording includes:

- “P4 and P8 are equivalent”;
- “P4 is better than P8”;
- “4/32 is the universal minimum interface”;
- “the original C61 is resolved”;
- cross-architecture minimum-dimension claims without a separate matched protocol.

## Provenance

- protocol lock commit: `1aad40d986ae5bf99bee72d7f57d03b23ce0b849`;
- successful implementation preflight run: `33646595085` using verification seed `58400`;
- fresh workflow run: `33646770156`;
- fresh workflow head SHA: `7106a7f68890600efdcdb66ac9d74cdd17ddc0a1`;
- raw fresh rows: `FRESH_ROWS.json`;
- locked aggregate: `DECISION.json`;
- artifact IDs and SHA256 digests: `RUN_MANIFEST.json`.
