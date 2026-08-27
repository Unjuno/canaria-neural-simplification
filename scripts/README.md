# Scripts

The active script surface is intentionally limited to code that is still needed to reproduce, replicate, audit, or review current evidence.

## Active paths

- `reproduce/` — supported clean-clone reproduction runners.
  - `reproduce/core_discovery_digits/` — representative residual-MLP direct experiment and full-cohort verifier.
  - `reproduce/g7_confirmatory/` — portable reproduction of an already-confirmatory G7 seed plus bounded runtime PoC.
- `replication/` — direct replication runners, including the SmallViT compositional experiment.
- `phase2/` — Phase 2 precision/quantization runners associated with the reviewed correction boundary.
- `phases/training_time/` — training-time evidence-producing runners still referenced by current reviewed results.
- `systems/` — S1–S7 streaming/native-runtime runners. Interpret them through `../docs/SYSTEMS_RUNTIME.md`.
- `recursive_composition/` — C-series exploration and confirmatory runners. Interpret them through `../docs/RECURSIVE_COMPOSITION.md`; script presence alone does not imply confirmatory status.

## Historical experiment code

Legacy versioned runners from the v11/v17–v23 sequence are preserved under:

`../archives/research-history/scripts/`

They may contain historical filesystem paths, dependency assumptions, and experimental interfaces. They are provenance, not a stable API.

Do not silently patch archived code and call the output an exact reproduction. Add a cleaned reproduction runner with explicit provenance instead.

## Reuse policy

- Protocol/result artifacts define what an evidence-producing run actually did.
- Prefer `reproduce/` when a supported reproduction path exists.
- Keep portability changes separate from scientific protocol changes.
- Reusable code belongs in `../src/canaria/` only when it can be separated from historical experiment semantics and tested independently.
- Preserve exploratory failures and correction records when they constrain interpretation; do not keep only positive outcomes.

## Reading order

1. `../REPOSITORY_LAYOUT.md`
2. `../docs/REPRODUCIBILITY.md`
3. the relevant evidence map (`CLAIMS_AND_EVIDENCE.md`, `RECURSIVE_COMPOSITION.md`, or `SYSTEMS_RUNTIME.md`)
4. the matching protocol/result artifact under `../results/`
5. the corresponding runner

For older version-number experiments, use `../archives/README.md` and the archived protocol/result/runner together.

## One-shot workflow policy

Experiment-specific GitHub Actions workflows used only to execute one exploration, confirmation, or reproduction check should be removed from the active branch after their output/provenance is retained. Git history preserves the execution scaffold.