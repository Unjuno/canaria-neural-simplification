# Scripts

The scripts in this repository come from an iterative research program rather than a pre-existing stable Python package. Directory role matters: some runners are intended for clean-clone reproduction, while others are preserved evidence-producing code with historical environment assumptions.

## Directory map

### `reproduce/` — supported clean-clone reproduction

Use this first when you want a runnable public reproduction path.

- `reproduce/core_discovery_digits/` — strongest minimal public residual-MLP direct experiment.
- `reproduce/g7_confirmatory/` — portable G7 seed reproduction and bounded runtime PoC.

These runners are the preferred place for additive portability fixes that do not alter the underlying scientific protocol.

### `replication/` — direct replication runners

Contains architecture/task replication code such as `replication/vit_compositional.py`. Read the matching locked protocol/result documents before interpreting outputs.

### `phase2/` — precision/quantization experiment runners

Contains the reviewed Phase 2 precision-composition code that corresponds to checked-in evidence. Phase 2 has an explicit correction boundary: read `../docs/phase2/README.md` and `../results/phase2/precision_composition/CORRECTION_STATUS.json` before using it.

### `phases/` — phase-specific / historical experiment code

Preserves evidence-producing runners from the longer research sequence. Some scripts contain historical filesystem or environment assumptions and are not presented as a stable API.

Do not equate a higher phase/version number with a stronger current public claim. Use `../docs/CLAIMS_AND_EVIDENCE.md` and `../results/README.md` for current interpretation.

## Reuse policy

- Phase protocols and result files are the authoritative description of each experiment.
- Historical scripts are preserved for auditability; some contain absolute `/mnt/data/...` paths from the original research environment.
- Do **not** silently edit a historical script and claim exact reproduction. For a portability fix, add a cleaned/versioned runner and record the change.
- The exact-codec implementation under `phases/v19/` is included primarily to expose the real bit-packing logic used for the 9,926-byte result. Its original dependency paths are intentionally visible.
- New reusable code should move into `../src/canaria/` only when its interface can be separated from a historical experiment without changing the evidence chain.

## Recommended order

1. Read `../REPOSITORY_LAYOUT.md`.
2. Read `../docs/REPRODUCIBILITY.md`.
3. Read the relevant phase protocol under `../docs/phases/`, `../docs/phase2/`, or historical protocol under `../docs/history/v10/`.
4. Check the matching JSON/CSV in `../results/`.
5. Prefer a runner under `reproduce/` when one exists.
6. Port historical paths/environment assumptions explicitly before rerunning older phase code on a new machine.

## One-shot workflow policy

Experiment-specific GitHub Actions workflows used only to execute one exploration/confirmation should be removed from the active branch after their outputs are committed and audited. Git history preserves the workflow that produced the evidence. Stable CI and intentionally supported reproduction workflows may remain under `.github/workflows/`.
