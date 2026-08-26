# Contributing

Canaria is maintained as an **auditable pre-announcement research repository**, not an open-ended benchmark race. Contributions are welcome when they improve reproducibility, portability, evidence quality, external validation, repository integrity, or a clearly scoped application proof-of-concept.

Read `REPOSITORY_LAYOUT.md`, `STATUS.md`, and `docs/ANNOUNCEMENT_READINESS.md` before making structural or claim-level changes.

## High-value contributions

- make a representative confirmatory experiment reproducible from a clean clone under a recorded environment;
- remove undocumented environment assumptions without rewriting historical evidence;
- add an independently motivated replication of compositional simplification;
- add stronger negative/null controls;
- improve statistical or accounting correctness;
- document a failed reproduction or contradictory result with enough detail to diagnose it;
- improve repository navigation, evidence indexing, and automated integrity checks without changing scientific claims.

## Research-change rules

1. **Do not overwrite historical experiment scripts or locked result files.** Add a new version/phase or a cleaned equivalent runner.
2. Declare whether a run is confirmatory, independent holdout, exploratory/pilot, reproduction, or secondary/mechanistic before interpreting the outcome.
3. For confirmatory work, define seed policy, eligibility, metric, threshold/decision rule, and stopping rule before fresh-seed outcome inspection.
4. Use matched continued-training controls when the claim concerns repair/recontracting over time.
5. Keep test data outside controller/endpoint-selection decisions unless a protocol explicitly declares otherwise; disclose weaker isolation designs.
6. Report negative results and protocol deviations.
7. Distinguish parameter count, optimizer-update proxies, FLOPs, wall-clock time, energy, nominal bits, entropy estimates, and actual serialized bytes.
8. Treat independently initialized training seed/model as the inferential unit for repeated within-model spans/checkpoints unless a stronger hierarchical analysis is supplied.
9. Preserve protocol/result hashes for major confirmatory phases when possible.
10. Do not convert an engineering hypothesis into a claimed systems result without measuring it directly.
11. If evidence is invalidated, preserve the original provenance and add an explicit correction record; do not silently delete the failed evidence chain.
12. A reproduction of already-observed seeds validates portability but does not increase the confirmatory scientific sample size.

## Branch and PR discipline

Keep scientific, reproduction, and maintenance changes separable:

- `main` is the reviewed evidence baseline during pre-announcement hardening.
- New experiments should use a dedicated research branch and normally a draft PR until the stopping rule is reached and evidence is reviewed.
- A successful experiment does not automatically change the claim registry; that requires a separate review/merge decision.
- Repository organization, CI maintenance, documentation cleanup, dependency pinning, and other science-neutral work should use a maintenance branch/PR and should not carry fresh experimental outcomes.
- Do not move or rewrite `v0.2.0-public-snapshot`.
- Do not describe a merge, CI PASS, or historical release as equivalent to announcement readiness; `docs/ANNOUNCEMENT_READINESS.md` is the current gate.

If a one-shot GitHub Actions workflow was created solely to execute an exploration, confirmation, reproduction audit, or maintenance action, remove it from the active branch after the result/provenance is retained. The producing workflow remains available in Git history. Stable CI and deliberately supported reproduction workflows may remain.

## Before proposing a new experimental branch

Read:

- `docs/CORE_DISCOVERY.md`
- `docs/CLAIMS_AND_EVIDENCE.md`
- `docs/NEGATIVE_RESULTS.md`
- `docs/OPEN_QUESTIONS.md`
- `docs/ROADMAP.md`
- `docs/ANNOUNCEMENT_READINESS.md`

Open an issue before large architectural changes or new theory-level/universality claims. A useful issue should identify the exact existing claim it tests, the competing explanation it distinguishes, the evidence class, and the minimum decisive experiment.

## Before merging

At minimum, run:

```bash
python -m unittest discover -s tests -v
python tools/audit_repo.py
python tools/audit_readiness.py
```

Also run any phase-specific evidence audit or reproduction command introduced by the PR. Confirm whether `docs/CLAIMS_AND_EVIDENCE.md` changes and that unmerged research is not presented as part of the reviewed baseline.