# Contributing

Begin with the [claim registry](docs/CLAIMS_AND_EVIDENCE.md), [Quickstart](QUICKSTART.md), and [research index](docs/RESEARCH_INDEX.md).

## Reproduction reports

Report exact commit, command, environment, CPU/device, seeds, expected and observed values. Attach machine-readable outputs and error logs without credentials or personal data. A mismatch is useful; do not silently replace failed seeds or adjust margins/tolerances to obtain PASS.

## Scientific changes

Use a research branch and an explicit experiment identifier. State hypothesis, controls, statistical unit, sampling/seed policy, thresholds, incomplete-cohort handling and stopping rule before fresh outcomes. Distinguish exploratory selection, fresh confirmation, post-hoc diagnostics, software reproduction and independent external replication. Preserve negative, uncertain and invalidated records.

Do not alter an old protocol or raw result to fit a new interpretation. Add a correction or follow-up. Fixed-split seed replication is not independent dataset replication. A correction subspace is not a complete model size or measured communication channel.

## Maintenance changes

Keep navigation/packaging separate from scientific outcomes. The preservation manifest checks original code/results and archived moves. If a real scientific correction is needed, document it and review a new version rather than relaxing the manifest silently.

Before submission:

```bash
python -m unittest discover -s tests -v
python tools/audit_repo.py
python tools/audit_publication.py
```

Run the full reproduction when changing its runner, numeric dependencies or interpretation. Do not submit generated files into original evidence paths. Do not add tokens, private datasets, credentials or environment secrets. CI is read-only and never needs a contributor's personal token.

Research PRs are not automatically merged or advertised. The owner selects the announcement scope explicitly, with unresolved limitations visible.
