# Results index

This directory contains compact machine-readable evidence for the public claims. It intentionally favors locked tables, summaries, decisions, paired effects, and seed metadata over generated plots or duplicated checkpoints.

## Directory map

- `history/v10/` — historical claim registry, experiment catalog, provenance maps, verified key results, and static audits.
- `phaseA_v11/` — blinded Stage-1 lock, locked composition table, Stage-3 join, cluster-bootstrap summaries, and confirmatory decision summary.
- `phaseB_v11/` — shell-capacity and head-capacity joint effect tables.
- `v12/` — equal-capacity location-control decisions and paired effects.
- `v13/` — float-budget and direct top-K weight summaries.
- `v14/` — precision × count sweep and repair summaries.
- `v15/` — quantizer, sparse-refit, bit-frontier, and FP16-scale results.
- `v16/` — structured-sparsity exploration and independent holdout.
- `v17/` — sub-100-byte core sequence, independent confirmation, and exact serialization evidence.
- `v18/` — global complexity accounting and whole-network low-bit summaries.
- `v19/` — head-compression sequence and exact 9,926-byte whole-network codec confirmation.
- `v20/` — small-ViT architecture-family generalization: confirmatory adapted-transfer and q8/zlib whole-network follow-up.

## How to interpret a result

1. Find the associated protocol/result narrative under `docs/phases/`.
2. Check whether the result is confirmatory, independent holdout, pilot, or exploratory.
3. Prefer seed/model-cluster uncertainty to treating multiple spans from the same network as independent samples.
4. For compression, distinguish:
   - core bytes;
   - nominal bit counts;
   - entropy/zlib proxy sizes;
   - real serialized whole-network bytes.
5. Do not exclude failed seeds or near-threshold cases after observing the outcome unless the exclusion rule was fixed beforehand.

## Canonical headline evidence

- Blind Phase A: `phaseA_v11/stage3_confirmatory_summary.json`
- Whole-network global accounting: `v18/phaseX_summary.json`
- Whole-network low-bit boundary: `v18/phaseY_summary.json`
- Exact 9,926-byte codec confirmation: `v19/confirmatory_codec_summary.json`
- Small-ViT adapted transfer: `v20/g3_confirmatory_summary.json`
- Small-ViT q8 whole-network follow-up: `v20/g3_q8_followup_summary.json`

Historical files are preserved as evidence, not necessarily as a clean modern API.
