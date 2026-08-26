## Purpose

What public-snapshot claim, reproducibility task, evidence artifact, bounded application task, or repository-maintenance task does this PR address?

## Change type

- [ ] maintenance / repository organization
- [ ] documentation / clarification
- [ ] reproducibility / portability
- [ ] historical evidence preservation
- [ ] new exploratory experiment
- [ ] new confirmatory / holdout experiment
- [ ] negative result
- [ ] deployment proof-of-concept
- [ ] reusable code / tests

## Baseline / branch boundary

- [ ] this PR is science-neutral maintenance, with no fresh experiment outcomes
- [ ] this PR contains research work and is isolated from the reviewed public baseline until review/merge
- [ ] the frozen `v0.2.0-public-snapshot` tag is untouched
- [ ] maintenance changes and fresh experimental outcomes are not mixed without an explicit reason

## Evidence integrity

- [ ] historical locked scripts/results were not overwritten
- [ ] evidence class was declared before interpreting new outcomes
- [ ] seed/checkpoint policy is documented
- [ ] test-set data is not used for controller/selection decisions, or the weaker isolation is explicitly disclosed
- [ ] matched continuation/control is used where the claim requires it
- [ ] negative outcomes/deviations are reported
- [ ] parameter count, serialized bytes, compiler updates, FLOPs, time, memory, and energy are not conflated
- [ ] invalidated evidence remains preserved as provenance rather than being silently removed

## Protocol / artifacts

List relevant paths and hashes where applicable:

- protocol/plan:
- code/script SHA256 or commit:
- result summary:
- artifact/result SHA256:

For maintenance-only PRs, describe the invariant being preserved instead.

## Workflow lifecycle

- [ ] no one-shot experiment workflow is being added
- [ ] one-shot workflow is still required for an active experiment
- [ ] completed one-shot workflows have been removed from the active branch after evidence/provenance was committed

## Claim impact

Does this PR change `docs/CLAIMS_AND_EVIDENCE.md`?

- [ ] no — maintenance / implementation / documentation / reproduction only
- [ ] yes — supported claim changes
- [ ] yes — rejected/boundary claim changes
- [ ] yes — open question changes

If yes, state the exact old and new claim wording. If no, confirm that new/unmerged research is not presented as part of the reviewed `main` baseline.

## Validation

```bash
python -m unittest discover -s tests -v
python tools/audit_repo.py
```

Report additional reproduction/experiment/evidence-audit commands as applicable.

## Scope limits

What does this PR **not** establish or change?
