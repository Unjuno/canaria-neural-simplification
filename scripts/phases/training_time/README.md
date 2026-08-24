# Training-time consolidation evidence scripts

These files preserve the evidence-producing scripts used in the G11/G15/G17 mechanism-separation series.

## Important provenance note

The scripts are intentionally kept close to the executed versions. Some imports use absolute `/mnt/data` paths because the experiments were run from a research handoff workspace containing historical v23–v25/G7/G10 modules. They are therefore **provenance artifacts, not yet clean standalone package entry points**.

Do not silently rewrite these files and then claim bitwise reproduction of the published runs. A portable refactor should be added separately under `src/canaria/` or a new script path, with equivalence tests against the preserved evidence scripts.

## Files

- `stable_auto_controller_v2.py` — calibration-only autonomous staged controller used as the stable G11/G15 baseline family.
- `direct_wait_control.py` — direct 4→2 candidate-tracking control used to test whether waiting for one large accurate merge matches staged consolidation.
- `factorized_compile_control.py` — 4→3→2 compiler-factorization control with no task learning between fits, used to separate algebraic factorization from recontracting.

See:

- `docs/TRAINING_TIME_CONSOLIDATION.md`
- `results/training_time/summary.json`
- `results/training_time/protocol_manifest.json`
