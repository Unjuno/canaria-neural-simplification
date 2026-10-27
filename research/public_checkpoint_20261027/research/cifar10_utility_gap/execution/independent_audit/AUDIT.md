> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Independent saved-output audit: fixed exploratory utility-gap v1

**PASS: 1015 checks, no discrepancies.** All 12 planned trajectories are complete and all 24 fixed checkpoints are present and hash-consistent. All 24 students have a descriptive teacher accuracy drop greater than 2 percentage points on the same 1,000 calibration diagnostic images: **2.1–6.4 pp**.

## Prespecified flags

- Primary width 8/MSE budget sensitivity: seed 26102603 only; fit NMSE reduction 10.005526868%, diagnostic NMSE reduction 10.083840090%, accuracy gain only 3/1000. Seeds 01/02 gains are 8/1000 each; neither meets both 10% NMSE thresholds. No budget contrast is task-relevant.
- Width 16 versus 8 at 2048 under MSE: NMSE sensitivity in all three seeds; correct-count gains 19/11/8. Task-relevant in seeds 01/02 only. No width benefit after apparent plateau flag. KL width contrasts do not meet both NMSE thresholds despite accuracy gains 24/1/10.
- KL versus MSE at 2048: no alignment evidence or alternative-objective-help flag. Correct-count changes at widths 8/16: seed 01 −19/−14, seed 02 −7/−17, seed 03 −6/−4. These are descriptive flags, not causal identifications.

## All fixed checkpoint outcomes

| Seed | Width / loss | Updates | Teacher correct | Student correct | Teacher drop (pp) |
|---|---|---:|---:|---:|---:|
| 26102601 | w8_mse | 1024 | 920 | 867 | 5.3 |
| 26102601 | w8_mse | 2048 | 920 | 875 | 4.5 |
| 26102601 | w8_kl | 1024 | 920 | 856 | 6.4 |
| 26102601 | w8_kl | 2048 | 920 | 856 | 6.4 |
| 26102601 | w16_mse | 1024 | 920 | 887 | 3.3 |
| 26102601 | w16_mse | 2048 | 920 | 894 | 2.6 |
| 26102601 | w16_kl | 1024 | 920 | 876 | 4.4 |
| 26102601 | w16_kl | 2048 | 920 | 880 | 4.0 |
| 26102602 | w8_mse | 1024 | 918 | 876 | 4.2 |
| 26102602 | w8_mse | 2048 | 918 | 884 | 3.4 |
| 26102602 | w8_kl | 1024 | 918 | 865 | 5.3 |
| 26102602 | w8_kl | 2048 | 918 | 877 | 4.1 |
| 26102602 | w16_mse | 1024 | 918 | 896 | 2.2 |
| 26102602 | w16_mse | 2048 | 918 | 895 | 2.3 |
| 26102602 | w16_kl | 1024 | 918 | 878 | 4.0 |
| 26102602 | w16_kl | 2048 | 918 | 878 | 4.0 |
| 26102603 | w8_mse | 1024 | 910 | 878 | 3.2 |
| 26102603 | w8_mse | 2048 | 910 | 881 | 2.9 |
| 26102603 | w8_kl | 1024 | 910 | 869 | 4.1 |
| 26102603 | w8_kl | 2048 | 910 | 875 | 3.5 |
| 26102603 | w16_mse | 1024 | 910 | 884 | 2.6 |
| 26102603 | w16_mse | 2048 | 910 | 889 | 2.1 |
| 26102603 | w16_kl | 1024 | 910 | 886 | 2.4 |
| 26102603 | w16_kl | 2048 | 910 | 885 | 2.5 |

## Arithmetic and lineage

- Recomputed 48 centered-NMSE values (24 fit, 24 diagnostic) from saved SSE/centered teacher SS/n and epsilon 1e-12×n×32×16×16. All match. Independently recomputed all 24 diagnostic mean KL values from logits using Python float64 stable log-softmax; maximum absolute deviation from production float32 values is 8.761626119e-9 (audit tolerance 1e-7).
- All 24 fixed checkpoint raw-byte hashes and fit-metric hashes match terminal and frozen-prediction receipts. Exactly 24 fixed checkpoint files exist; rotating last-good files are recovery artifacts, not additional fixed-budget outcomes. Fresh-Adam/no-warm-start and equal objective-pair initialization hashes agree in session metadata; optimizer tensors were not loaded.
- Original teacher checkpoint raw-byte hashes still match pinned identities. Recorded state before/after hashes are equal in all three seeds and match session and fit metric bindings. Original result metadata files retain their locked hashes and negative-result declarations.

## Interpretation and limits

- Saved-output arithmetic and artifact-binding audit only; no training or inference rerun, model/optimizer/checkpoint deserialization, image/label-payload loading, original-validation/test/archive access, network call, or CI. Checkpoint files were opaque SHA256 byte-streams only.
- Centered NMSE was recomputed from saved SSE, centered teacher SS, n and fixed 32x16x16 features; underlying endpoint tensors are not retained in these outputs.
- Diagnostic KL recomputed independently in Python float64 from saved per-example logits; production float32 arithmetic permits absolute tolerance 1e-7. Fit KL has no saved per-example logits/sums here, so only its finite/nonnegative value and hash binding can be verified, not independently reproduced. Uncentered energy NMSE cannot be independently reproduced because teacher energy is not saved.
- Teacher before/after state hashes match recorded metadata; no independent tensor-state rehash was performed (would require prohibited weight deserialization). Current original teacher checkpoint raw byte hashes match pinned identities.
- One-look lineage is consistent across exclusive-marker source, canonical hashes, process receipts, ledger and filesystem ordering. Saved evidence alone is not an adversarial proof of absence of unlogged access.
- All >2 percentage-point drops are descriptive on shared 1000 calibration diagnostics, not formal noninferiority, independent seed/image replications, pristine study holdout, or original heldout failure replacement.

The ingestion amendment explicitly permits transient TRAIN mixed-batch decoding only to retain fixed calibration rows; original validation is not used, and official test is not opened according to the bound source/receipts. This audit does not reopen any data array or archive.

## Reproducibility evidence

- `audit_saved_results.py`: independent single-thread stdlib audit; does not call production calculations.
- `audit_result.json`: every check plus complete contrast recomputation and scope limitations.
- `recomputed_all 24.csv`: all exact paired counts, drops, NMSEs and diagnostic KL values.
- `input_hashes.json`: every input file hashed by this audit.
- `audit_stdout.log`: run summary.
- `audit_saved_results.schema_error_preserved.py`: preserved initial audit-script schema mistake (`cohort` versus actual `cohort_clock`), corrected before successful audit. This was an audit-only parsing failure, not a scientific rerun.

Report SHA256: `60cb3fde0437c352999b1a818c88aeb8316524bbb3361665414e8f2b44930cac`.
Frozen predictions SHA256: `0ac9b67457133962b4a25db56c3bdfa4d1411c9ca087d867e24225284ee569ec`.
