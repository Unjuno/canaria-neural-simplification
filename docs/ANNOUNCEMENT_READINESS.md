# Announcement readiness — scoped research preview

Issue #13 is the readiness tracker; Issue #16 is the historical-layout tracker. Public visibility and the frozen `v0.2.0-public-snapshot` do not themselves pass this gate.

## Current outcome: BLOCKED

The full-cohort numerical reproduction failed in the modern pinned environment and in a local previous-version stack. [Issue87 / evidence](REPRODUCTION_DISCREPANCY.md). Directional agreement does not clear the strict reproduction requirement. Integrity checks may pass while this scientific/readiness gate remains blocked.

## Intended announcement

Release **research code and an auditable, bounded evidence collection**, with one reproducible baseline as the headline. Do not announce a completed universal compression technology. Newer results are explicitly linked research, not silently incorporated as new reviewed-baseline claims.

## Gate

| Requirement | Required evidence |
|---|---|
| Scope is explicit | README, STATUS, claim registry and publication policy agree; other scientific families are explicitly included as supporting history or excluded from the headline. |
| Clean reproduction | Entire recorded core cohort1200–1207 rerun under the pinned Python3.11/CPU environment; exact endpoint comparison and numerical-statistic checks; failures retained. |
| Preservation | Baseline scientific files and migrated historical files match recorded hashes; frozen tag unchanged. |
| Navigability | Current Markdown local links resolve; legacy material is under archives with an explicit old-to-new map. |
| Integrity | Unit tests, repository audit, publication audit, and final-commit CI succeed. |
| Honesty | No claim that same-code reruns equal external replication/peer review; uncertainty, missing raw artifacts, and invalidations remain disclosed. |

The executable gate is `python tools/audit_publication.py`. With `--require-reproduction`, it additionally requires a persisted PASS report whose input hashes match the current scientific files. The gate fails closed for missing evidence. Workflow artifacts alone must be preserved before advertising closure.

## Not launch blockers for this limited preview

A new dataset, CNN/Transformer rerun, exact minimal-dimension theory, or real-device performance is not needed **because none is asserted by the headline**. These would be required for their respective stronger claims. Independent external reproduction and scientific peer review remain outstanding and are not replaced by this gate.

## Maintenance/review scope

This change preserves the already reviewed scientific baseline and narrows communication. It does not approve the scientific contents of every open PR, close unresolved experiments, retune thresholds, or merge the large research stack. Older maintenance PRs #14/#17 can be superseded only after their environment/layout requirements are demonstrably covered; their history remains intact.

No social post, release tag, DOI, or peer-reviewed publication is created by a PASS. The owner can announce the scoped research preview after the final branch/main checks and evidence persistence are complete. See the current [reproduction record](../results/reproduction/README.md), [status](../STATUS.md), and [scope policy](../publication/CLAIM_POLICY.json).
