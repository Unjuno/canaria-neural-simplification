# C66R result — Gaussian shift severity frontier exploration

## Status

**PROSPECTIVE EXPLORATORY**. Confirmatory claim is not allowed from C66R.

Protocol was locked before outcomes at commit `12c33e2decf5ff1358d0058257ba67aefed24eb6`. Fresh model seeds were `64400–64415`; all 16 were eligible. Held-out test data were not used.

## Terminal decision

`NO_P0_FAILURE_THROUGH_SIGMA_0_20`

Under the locked Residual-MLP protocol, P0 jointly passed the exploratory validation non-inferiority and NMSE-ratio gates at every tested Gaussian severity: `0.04, 0.08, 0.12, 0.16, 0.20`.

| sigma | P0-P2 validation mean (pp) | bootstrap 95% (pp) | P0/P2 NMSE geomean | bootstrap 95% | joint |
|---:|---:|---:|---:|---:|---|
| 0.04 | -0.324074 | [-0.601852, -0.023148] | 1.010180 | [0.996580, 1.023684] | PASS |
| 0.08 | -0.324074 | [-0.671296, 0.046297] | 1.012235 | [0.997840, 1.027727] | PASS |
| 0.12 | -0.185186 | [-0.416668, 0.046295] | 1.019922 | [1.003492, 1.037779] | PASS |
| 0.16 | -0.138888 | [-0.578703, 0.300927] | 1.024717 | [1.006466, 1.042604] | PASS |
| 0.20 | -0.162037 | [-0.671297, 0.347222] | 1.025869 | [1.006195, 1.045037] | PASS |

Locked margins were `-2 pp` for validation non-inferiority and `1.25` for the NMSE ratio. Bootstrap used 100,000 paired model-seed resamples with RNG seed `4017607924` and a common bootstrap index matrix across severity.

## Mechanism observations

These diagnostics are descriptive, not confirmatory gates.

- Mean shifted-teacher validation accuracy declined from `0.9773` at sigma `0.04` to `0.9171` at sigma `0.20`.
- Mean shifted-teacher accuracy drop relative to clean increased from about `-0.139 pp` to `-6.157 pp`.
- Mean frozen-hierarchy activation NMSE versus teacher increased from `0.07270` to `0.12564`.
- P2 Euclidean residual capture stayed near `0.23–0.24` on average rather than increasing sharply.
- P2 logit-L2 retained ratio declined modestly from about `0.722` to `0.702`.

Thus shift severity clearly degrades the teacher and increases frozen-hierarchy discrepancy, but within this grid it does **not** produce a locked P0-versus-P2 failure.

## Interpretation boundary

Safe statement:

> In the repository Residual-MLP testbed, the C66R exploratory common-random-number Gaussian severity grid found no P0-versus-P2 non-inferiority failure through sigma `0.20` under the locked margins.

Not supported:

- sigma `0.20` is a confirmed robustness limit;
- P0 is universally sufficient through sigma `0.20`;
- no failure exists above sigma `0.20`;
- the exact continuous critical sigma exceeds `0.20` under arbitrary Gaussian shifts;
- P2 is useless or teacher correction is universally unnecessary;
- any conclusion about the imported Residual CNN C59/C60 line.

Because no grid point failed, C66R selected **no** sigma for a confirmatory C67R. The next valid experiment is a separately locked exploratory extension above sigma `0.20`, with safeguards ensuring that teacher/reference degradation does not make P0-versus-P2 non-inferiority uninterpretable.
