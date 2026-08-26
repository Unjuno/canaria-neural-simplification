> **Historical handoff — completed.** The independent re-review described below was completed on 2026-08-26. This file is preserved at its original path for provenance and is **not current project instruction**. The active pre-announcement gate is `docs/ANNOUNCEMENT_READINESS.md` / Issue #13.

# Independent re-review handoff

This repository is at a **pre-publication independent re-review gate**.

The next review session should not extend the research by default. Its job is to decide, claim by claim, what is safe to keep in the public-facing repository.

## Required output of the re-review

For every material public claim, assign one of:

- **KEEP** — directly supported by the cited protocol/result files and stated within scope.
- **EDIT** — evidence is usable but wording, statistics, scope, or provenance needs correction.
- **REMOVE** — not adequately supported for a public-facing claim. Remove it from README/STATUS/public docs.
- **INVALIDATE** — experiment or inference is technically invalid. Preserve the invalidated artifact/history for audit, but do not present it as evidence.

Do not silently delete failed or invalidated raw evidence. Remove incorrect statements from the public surface; preserve provenance under results/history with an explicit invalidation note.

## Review order

### 1. Public surface first

Review these before reading the full history:

1. `README.md`
2. `QUICKSTART.md`
3. `STATUS.md`
4. `docs/CLAIMS_AND_EVIDENCE.md`
5. `docs/phase2/README.md`

Question: **Would a reader who only sees these files leave with any unsupported or misleading belief?**

### 2. Core empirical claim

Check the component-wise vs composed simplification claim against:

- `results/core_discovery_digits/PROTOCOL_LOCK.json`
- `results/core_discovery_digits/confirm_summary.json`
- `scripts/reproduce/core_discovery_digits/run_confirmatory.py`
- `results/replication/vit_compositional/PROTOCOL_LOCK.json`
- `results/replication/vit_compositional/confirm_summary.json`
- `docs/CORE_DISCOVERY_REPLICATION_DIGITS.md`
- `docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md`

Audit specifically:

- validation/test separation;
- equal learned-parameter accounting where claimed;
- seed independence and selection rules;
- bootstrap/sign/Wilcoxon statements;
- whether `composition` is being confused with architecture change;
- whether operational complexity is being overstated as mathematical/Kolmogorov complexity.

### 3. Training-time claims

Review:

- `docs/TRAINING_TIME_CONSOLIDATION.md`
- `docs/LATE_STAGE_FINDINGS.md`
- `results/training_time/summary.json`
- `results/training_time/late_stage_summary.json`
- `docs/NEGATIVE_RESULTS.md`

Audit whether exploratory controller/mechanism findings are clearly separated from confirmatory claims.

### 4. Phase 2 precision/quantization claims

Start from the authoritative correction index:

- `docs/phase2/README.md`
- `results/phase2/precision_composition/CORRECTION_STATUS.json`

Then inspect Phase 2A–C protocol/result files and portable runners.

Critical known correction:

- Phase 2E is `INVALIDATED_IMPLEMENTATION_BUG` because repair used raw `Xt` instead of the internal activation domain `ta[0]`; the equal 64-dimensional width made the bug silent.
- Any public statement whose causal interpretation depended on the Phase 2E failure must be removed or rewritten.
- Corrected later work supports viability of short activation-domain repair for coarse 3-bit per-matrix quantization in the tested residual-MLP family.
- A lower QAT repair sample complexity for the composed condition is **not confirmed**; Phase 2O was `UNCERTAIN`.

### 5. Reproducibility and systems boundary

Review:

- `scripts/reproduce/g7_confirmatory/`
- `results/reproduction/g7_seed4300_report.json`
- `docs/RUNTIME_POC.md`
- `results/reproduction/runtime_poc_seed4300_report.json`

Check that exact reproduction is not described as an independent scientific replication, and that CPU/storage measurements are not generalized to GPU/RAM/energy or large models.

## Minimal code audit

At minimum, inspect the public runners for:

- wrong activation domains;
- shape coincidences that can hide semantic errors;
- train/validation/test leakage;
- post-hoc selection presented as locked selection;
- mismatched parameter or coded-size accounting;
- incorrect quantizer metadata accounting;
- local `/mnt/data` dependencies in files presented as portable.

Run `python tools/audit_repo.py` after any edits.

## Minimal experiment rule

Do **not** start another broad experiment family during review.

A new experiment is justified only if all three are true:

1. the reviewer finds a concrete public claim whose validity cannot be decided from existing evidence;
2. a small, prelocked experiment can decide whether to KEEP or REMOVE that claim;
3. removing the claim would otherwise discard a central result rather than a secondary detail.

If those conditions are not met, prefer **removing or narrowing the claim** over adding experiments.

## Publication gate

Publication/posting should happen only after:

- [ ] public-surface claims reviewed;
- [ ] Phase 2E correction propagated everywhere relevant;
- [ ] unsupported wording removed;
- [ ] invalidated evidence clearly labeled rather than hidden;
- [ ] minimal public runner smoke-tested;
- [ ] `repository-audit` passes;
- [ ] Draft PR #7 reviewed after the v0.2.0 release boundary is handled;
- [ ] independent re-review issue is closed.

## Reviewer note

The goal is not to make the repository look maximally impressive. The goal is to leave a small number of claims that another person can inspect, reproduce, and extend without inheriting hidden assumptions.
