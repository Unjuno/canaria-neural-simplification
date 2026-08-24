# Project status

**2026-08-24: external-validity experiments resumed.** The repository remains the canonical research record, but the project is now actively executing the cross-architecture generalization roadmap.

## Current experimental frontier

The v20-v23 generalization series now contains both positive and negative transfer:

- **v20 / G3:** residual CNN -> small ViT: **A — adapted transfer**.
- **v21 / G5:** image-token ViT -> non-image Transformer encoder: **Z — zero-shot transfer**.
- **v22 / G6:** Transformer encoder -> synthetic causal decoder: **A — adapted transfer**; PPL-only evaluation was insufficient because zero-shot generation drifted.
- **v23 / G6b:** synthetic causal language -> held-out natural English character LM: **N — no transfer under tested budget**. Teacher-forced PPL remained near-identical at tau0 while autoregressive rollout fidelity failed; the prespecified bounded joint-repair adaptation also failed.

The current objective is therefore not to accumulate positive compression results. It is to map **where simplification transfers, where architecture/task-specific adaptation is sufficient, and where autoregressive/error-amplification dynamics create a genuine applicability boundary**.

## Repository policy during active experiments

- Freeze protocols/conditions before confirmatory cohorts.
- Keep pilot, exploratory, confirmatory, independent-holdout, and negative evidence explicitly separated.
- Never weaken a confirmatory threshold because a pilot or early seed is difficult.
- Use matched continued-training controls for repair experiments.
- Use seed/model as the inference unit when repeated events share a trained model.
- Preserve failed hypotheses, non-replications, runtime amendments, and ineligible baseline seeds.
- Distinguish core bytes, parameter/state-stream bytes, entropy/code proxies, and real standalone serialized whole-model bytes.
- For autoregressive models, teacher-forced likelihood is not sufficient; rollout-sensitive metrics remain required.
- Historical evidence-producing scripts should be preserved; portability fixes belong in additive modern modules/scripts.

See `docs/GENERALIZATION_STATUS.md` for the live transfer map and `docs/CLAIMS_AND_EVIDENCE.md` for the current claim registry.
