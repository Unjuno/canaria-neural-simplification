# Contributing

Canaria is currently maintained as an **auditable research snapshot**, not an open-ended benchmark race. Contributions are welcome when they improve reproducibility, portability, evidence quality, external validation, or a clearly scoped application proof-of-concept.

## High-value contributions

- make a representative confirmatory experiment runnable from a clean clone;
- remove undocumented environment assumptions without rewriting historical evidence;
- add an independently motivated replication of compositional simplification;
- add stronger negative/null controls;
- improve statistical or accounting correctness;
- implement a minimal runtime-compilation/materialization proof-of-concept;
- document a failed reproduction or contradictory result with enough detail to diagnose it.

## Research-change rules

1. **Do not overwrite historical experiment scripts or locked result files.** Add a new version/phase or a cleaned equivalent runner.
2. Declare whether a run is confirmatory, independent holdout, exploratory/pilot, or a reproduction attempt before interpreting the outcome.
3. For confirmatory work, define seed policy, eligibility, metric, threshold/decision rule, and stopping rule before fresh-seed outcome inspection.
4. Use matched continued-training controls when the claim concerns repair/recontracting over time.
5. Keep test data outside controller commit decisions.
6. Report negative results and protocol deviations.
7. Distinguish parameter count, optimizer-update proxies, FLOPs, wall-clock time, energy, nominal bits, entropy estimates, and actual serialized bytes.
8. Treat independently initialized training seed/model as the inferential unit for repeated within-model spans/checkpoints unless a stronger hierarchical analysis is supplied.
9. Preserve protocol/result hashes for major confirmatory phases when possible.
10. Do not convert an engineering hypothesis (for example runtime memory savings) into a claimed result without measuring it directly.

## Before proposing a new experimental branch

Read:

- `docs/CORE_DISCOVERY.md`
- `docs/CLAIMS_AND_EVIDENCE.md`
- `docs/NEGATIVE_RESULTS.md`
- `docs/OPEN_QUESTIONS.md`
- `docs/ROADMAP.md`

Open an issue before large architectural changes or new theory-level/universality claims. A useful issue should identify the exact existing claim it tests, the competing explanation it distinguishes, and the minimum decisive experiment.
