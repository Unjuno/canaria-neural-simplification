# Phase scripts

This directory contains **selected original research scripts** for later phases. They are preserved close to the form used to generate the published result files; the public repository does not pretend that every historical script has already been ported into a clean API.

## Reproduction order

1. Read the matching protocol under `docs/phases/`.
2. Check `docs/REPRODUCIBILITY.md` for seed, blindness, eligibility, and matched-control rules.
3. Run phase scripts in an isolated output directory; never overwrite locked result files.
4. Compare regenerated summaries with the corresponding files under `results/`.

## Historical-script caveat

Some original scripts contain absolute `/mnt/data/...` paths or imports into the original experiment workspace. They are provenance artifacts, not portable entry points. Portability fixes should be added as new scripts/modules rather than silently rewriting evidence-producing historical scripts.

## Current public script coverage

- `v11/` — blinded Phase-A Stage-1 implementation.
- `v17/` — 44.5 B independent confirmation and exact ternary/enumerative core codecs.
- `v18/` — whole-network low-bit evaluation; Phase-X accounting protocol/results remain public even where the original monolithic script is not yet ported.
- `v19/` — exact 9,926-byte whole-network pack/unpack implementation.
- `v20/` — small-ViT architecture-generalization runner.
- `v21/` — non-image Transformer-encoder zero-shot transfer runner.
- `v22/` — causal decoder-LM confirmatory and q8 follow-up runners.
- `v23/` — real-text character-LM model/data/evaluation utilities and confirmatory negative-transfer runner.

The v13–v16 protocols/results are public under `docs/phases/` and `results/`; their original experiment scripts remain historical/non-portable unless explicitly added later.

## Autoregressive phases

For v22 and later decoder experiments, **teacher-forced PPL is not accepted as sufficient functional evidence**. Reproduction should retain the preregistered free-running rollout metrics. v23 is intentionally a negative result: the model preserves PPL while failing rollout fidelity under the tested compiler/adaptation budget.

## Portable reusable code

For new code, prefer the dependency-light modules under `src/canaria/`. In particular, `src/canaria/ternary_codec.py` extracts the v17 fixed ternary and zero-aware enumerative codecs from the training workspace. These functions are exercised by `tests/test_ternary_codec.py` in CI.

The exact model-specific whole-network pack/unpack implementation remains `v19/run_phaseAD_exact_codec_v19.py`.
