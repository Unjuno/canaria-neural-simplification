# Phase scripts

This directory contains selected original research scripts for the later phases. They are preserved close to the form used to generate the published result files.

## Reproduction order

1. Read the matching protocol under `docs/phases/`.
2. Check `docs/REPRODUCIBILITY.md` for seed and blindness rules.
3. Run the phase script in an isolated output directory; never overwrite locked result files.
4. Compare regenerated summaries with the corresponding files under `results/`.

## Important caveat

These are research scripts, not yet a stable library API. Some historical helpers assumed the original filesystem layout. Portability fixes should be added as new scripts/modules rather than silently rewriting evidence-producing historical scripts.

## Publicly prioritized implementations

- v11 — blinded Phase-A Stage 1 / chunk execution.
- v14 — precision × weight-count sweep.
- v15 — quantizer/sparse-refit/storage-frontier experiments.
- v16 — structured-sparsity and independent holdout.
- v17 — independently confirmed sub-100-byte core and exact core codecs.
- v18 — whole-network complexity accounting and low-bit evaluation.
- v19 — head compression and exact 9,926-byte codec.

The exact whole-network pack/unpack implementation is `v19/run_phaseAD_exact_codec_v19.py`.