# Project status

**2026-08-24: corrected mainline is training-time functional consolidation and autonomous self-recompilation.**

The repository remains the canonical research record. Earlier post-hoc/generalization experiments are preserved, including the v23–v25 natural-text failures. The current mainline was corrected after an internal audit showed that local deletion/pruning experiments were drifting away from the original Canaria question of consolidating a wider learned computation into a smaller mechanism and then allowing the surrounding network to adapt.

## Current mainline

Working process:

> **form → transfer → commit → recontract → transfer again**

A larger model first forms useful computation. A span is transferred into a smaller replacement; the replacement is committed before perfect equivalence is required; task learning resumes; later consolidations repeat the process.

### Confirmed training-time results

- **G7:** progressive `4→3→2` consolidation at the same final 2-block / MLP24 capacity beat early and late one-shot `4→2` schedules on fresh seeds 4300–4307. Final reduction from the 4-block / MLP48 reference: **52.28% parameters**.
- **G8:** correct function-aligned transfer was required under identity and shuffled-target controls.
- **G9:** more accurate functional transfer improved final task utility, with clear diminishing returns as compiler-fit budget increased.
- **G10:** structured weight inheritance alone was insufficient; inheritance followed by function-aligned refinement outperformed random-initialized functional fitting.
- **G11:** the calibration-only autonomous controller reached the final 2-block architecture in **8/8** fresh seeds while satisfying the locked +2% non-inferiority criterion against the Large reference; maximum compiler updates were **192** versus the preregistered maximum **232**.
- **G15:** staged `4→3→2` with task learning between commits beat waiting for a direct `4→2` merge by **−0.2993 PPL**, 95% CI **[−0.3379, −0.2616]**, 8/8 fresh seeds.
- **G17:** when task learning was removed between `4→3` and `3→2`, the factorized path was equivalent to direct `4→2` within the preregistered ±0.10 PPL band. This isolates the current strongest mechanism: **the staged benefit depends on intervening task learning/recontracting, not merely on splitting one compiler fit into two fits.**

## Historical boundary result retained

The v23–v25 natural-English character-LM series remains a valid negative result for the tested post-hoc/bounded-repair regime: teacher-forced likelihood could remain near-identical while free-running autoregressive trajectories diverged. The later training-time results do not erase this result; they change the intervention path being studied.

## Evidence policy

- Freeze confirmatory protocols before inspecting fresh-seed outcomes.
- Keep exploratory, confirmatory, independent-holdout, equivalence, and negative evidence explicitly separated.
- Do not promote runs with broken/missing protocol-lock integrity to confirmatory status.
- Use independently initialized model seed as the inference unit unless a different unit is preregistered.
- Keep test data outside controller commit decisions; use training/calibration data only for autonomous consolidation choices.
- Preserve failed controls and mechanism-separation experiments.
- Distinguish parameter count, compiler-update proxy, exact FLOPs, wall clock, energy, compressed state-stream bytes, and standalone serialized bytes.
- Do not equate teacher fidelity with task utility.

## Current execution frontier

The highest-information next test is **G18: a recontracting-aware autonomous policy**. The stable controller currently uses a static functional-NMSE threshold. G13–G17 show that commit quality depends not only on instantaneous replacement error but also on the amount of task-learning horizon remaining after the commit.

A second required test is **G19: staged-path generalization** to a different source depth/path (for example `5→4→3→2` versus `5→2`) before treating the staged effect as architecture-independent.

See:

- `docs/TRAINING_TIME_CONSOLIDATION.md`
- `docs/NEXT_EXPERIMENTS_AUTONOMOUS.md`
- `results/training_time/summary.json`
- `results/training_time/protocol_manifest.json`
- `scripts/phases/training_time/stable_auto_controller_v2.py`
