## Purpose

What public-snapshot claim, reproducibility task, evidence artifact, or bounded application task does this PR address?

## Change type

- [ ] documentation / clarification
- [ ] reproducibility / portability
- [ ] historical evidence preservation
- [ ] new exploratory experiment
- [ ] new confirmatory / holdout experiment
- [ ] negative result
- [ ] deployment proof-of-concept
- [ ] reusable code / tests

## Evidence integrity

- [ ] historical locked scripts/results were not overwritten
- [ ] evidence class was declared before interpreting new outcomes
- [ ] seed/checkpoint policy is documented
- [ ] test-set data is not used for controller/selection decisions
- [ ] matched continuation/control is used where the claim requires it
- [ ] negative outcomes/deviations are reported
- [ ] parameter count, serialized bytes, compiler updates, FLOPs, time, memory, and energy are not conflated

## Protocol / artifacts

List relevant paths and hashes:

- protocol/plan:
- code/script SHA256:
- result summary:
- artifact/result SHA256:

## Claim impact

Does this PR change `docs/CLAIMS_AND_EVIDENCE.md`?

- [ ] no — implementation/documentation/reproduction only
- [ ] yes — supported claim changes
- [ ] yes — rejected/boundary claim changes
- [ ] yes — open question changes

If yes, state the exact old and new claim wording.

## Validation

```bash
python -m unittest discover -s tests -v
python tools/audit_repo.py
```

Report additional reproduction/experiment commands as applicable.

## Scope limits

What does this PR **not** establish?
