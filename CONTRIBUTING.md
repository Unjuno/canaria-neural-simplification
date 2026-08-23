# Contributing

Contributions that improve reproducibility, portability, statistical correctness, or external validation are welcome.

## Rules for research changes

1. Do not overwrite historical experiment scripts or locked result files. Add a new phase/version.
2. State whether a new experiment is confirmatory, independent holdout, pilot, or exploratory **before** interpreting its results.
3. Define the seed policy, eligibility rule, metric, threshold, and stopping rule before outcome inspection for confirmatory work.
4. Use matched continued-training controls for repair/adaptation experiments.
5. Report negative results.
6. Distinguish nominal bit counts, entropy estimates, and actual serialized file sizes.
7. For repeated spans/events within one model, treat the training seed/model as the inferential cluster unless a stronger hierarchical analysis is supplied.
8. Include a short protocol Markdown file next to each major new phase.

Please open an issue before large architectural changes or new theory-level claims.
