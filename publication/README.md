# Publication metadata

`CLAIM_POLICY.json` defines editorial inclusion, not scientific outcomes. `PRESERVATION_MANIFEST.json` checks the original evidence/code and byte-identical migrations. `PATH_MIGRATION.json` resolves archived paths. These are preparation-time records and cannot be cited as original preregistration.

The executable check is `python tools/audit_publication.py`; the numerical reproduction is a separate workflow.

## Evidence-policy scope after the October checkpoint

This folder holds the historical baseline's policy/catalog and immutable preservation metadata. The October Digits L/J/C and CIFAR reports live on a **separate research checkpoint branch** and are not promoted merely by linking them. The September R87R2/Phase4 [publication candidate PR #101](https://github.com/Unjuno/canaria-neural-simplification/pull/101) and October [documentation PR #106](https://github.com/Unjuno/canaria-neural-simplification/pull/106) have different bases and scope. Read [branch/evidence operations](../docs/REPOSITORY_OPERATIONS.md) before changing a policy file.

A publication decision requires independent claim selection, unchanged original evidence, exact-head audit and a separate merge/release decision; current Issue #87 historical exact-reproduction debt is not relabeled solved by a new cohort.
