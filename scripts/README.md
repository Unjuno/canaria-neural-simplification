# Scripts

The scripts in this repository come from an iterative research program rather than a pre-existing stable Python package.

## Reuse policy

- Phase protocols and result files are the authoritative description of each experiment.
- Historical scripts are preserved for auditability; some contain absolute `/mnt/data/...` paths from the original research environment.
- Do **not** silently edit a historical script and claim exact reproduction. For a portability fix, copy it to a new path/version and record the change.
- The exact-codec implementation under `scripts/phases/v19/` is included primarily to expose the real bit-packing logic used for the 9,926-byte result. Its original dependency paths are intentionally visible.

## Recommended order

1. Read `docs/REPRODUCIBILITY.md`.
2. Read the relevant phase protocol under `docs/phases/` or historical protocol under `docs/history/v10/`.
3. Check the matching JSON/CSV in `results/`.
4. Port paths/environment assumptions explicitly before rerunning on a new machine.

A future library-quality refactor should be additive (`src/canaria/` or a new phase) and should not rewrite the historical evidence chain.
