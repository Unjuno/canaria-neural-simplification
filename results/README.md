# Results index

This directory contains machine-readable research evidence. It intentionally favors locked protocols, summaries, decisions, paired effects, seed metadata, and correction records over generated plots or duplicated checkpoints.

**Do not infer current public-claim status from a directory version number alone.** Current interpretation is controlled by `../docs/CLAIMS_AND_EVIDENCE.md`, `../STATUS.md`, and the relevant review/correction documents.

## 1. Reviewed public headline evidence

These paths are the shortest route from the current reviewed public claims to machine-readable evidence:

- `core_discovery_digits/` — residual-MLP direct component-wise/composed replication with exact learned replacement-parameter matching.
  - `PROTOCOL_LOCK.json`
  - `confirm_summary.json`
- `replication/vit_compositional/` — SmallViT direct replication and seed table.
- `training_time/` — consolidated G7–G27 training-time protocol/summary material.
- `phase2/precision_composition/` — reviewed Phase 2A–C precision evidence plus explicit correction/invalidation records.

For Phase 2, read `phase2/precision_composition/CORRECTION_STATUS.json` and `INVALIDATED_HISTORY.md` before using later precision-composition material. Invalidated evidence is preserved for provenance and is not scientific support.

## 2. Reproduction / systems evidence

- `reproduction/g7_seed4300_report.json` — portable exact reproduction of an already-confirmatory G7 seed; portability evidence, not an independent scientific replication.
- `reproduction/runtime_poc_seed4300_report.json` — bounded small CPU serialization/direct-execution PoC.

Read `reproduction/README.md` and the corresponding documentation before generalizing either result.

## 3. Extended phase sequence and historical evidence

The following directories preserve the longer research trajectory and specialized compression/generalization phases:

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
- `v20/` — small-ViT architecture-family generalization: adapted-transfer confirmatory and q8/zlib follow-up.
- `v21/` — non-image Transformer encoder zero-shot transfer and q8 follow-up.
- `v22/` — causal decoder LM adapted transfer with paired PPL / free-generation metrics and q8 follow-up.
- `v23/` — negative real-text causal-LM generalization; PPL preserved while autoregressive rollout fidelity fails under both zero-shot and the prespecified bounded adaptation.
- `v24/` and `v25/` — later real-text causal-LM diagnostics/adaptation variants; interpret through their phase documents rather than treating higher version numbers as stronger public claims.

Detailed protocol/result narratives for this sequence live under `../docs/phases/`.

## 4. Evidence immutability and correction policy

Once a result artifact has been used for confirmatory inference or a public claim, do not silently replace its contents. If a bug, provenance problem, or interpretation error is found:

1. preserve the original artifact where feasible;
2. add a correction/invalidation record;
3. remove or narrow dependent public claims;
4. rerun under a new corrected protocol/version if scientific re-evaluation is needed.

A result can therefore be historically important while being invalid for inference.

## 5. How to interpret a result

1. Find the associated protocol/result narrative under `../docs/`, especially `../docs/phases/` for the versioned historical sequence.
2. Check whether the result is confirmatory, independent holdout, pilot/exploratory, reproduction, secondary/mechanistic, uncertain, or invalidated.
3. Prefer seed/model-cluster uncertainty to treating multiple spans from the same network as independent samples.
4. For compression, distinguish core bytes, nominal bit counts, entropy/zlib proxy sizes, state-stream bytes, and real standalone serialized whole-network bytes.
5. Do not exclude failed seeds or near-threshold cases after observing the outcome unless the exclusion rule was fixed beforehand.
6. For autoregressive models, do not treat teacher-forced PPL as sufficient evidence of successful functional transfer; check rollout-sensitive metrics.
7. Check correction records before using a result in a new claim.

## Canonical historical headline files

These remain useful entry points into the extended phase sequence:

- Blind Phase A: `phaseA_v11/stage3_confirmatory_summary.json`
- Whole-network global accounting: `v18/raw/phaseX_summary.json`
- Whole-network low-bit boundary: `v18/raw_phaseY/phaseY_summary.json`
- Exact 9,926-byte codec confirmation: `v19/raw_AD/confirmatory_codec_summary.json`
- Small-ViT adapted transfer: `v20/g3_confirmatory_summary.json`
- Sequence-Transformer zero-shot transfer: `v21/g5_confirmatory_summary.json`
- Synthetic causal-decoder adapted transfer: `v22/g6_confirmatory_summary.json`
- Real-text causal-decoder negative transfer: `v23/g6b_confirmatory_summary.json`

Historical files are preserved as evidence, not necessarily as a clean modern API.
