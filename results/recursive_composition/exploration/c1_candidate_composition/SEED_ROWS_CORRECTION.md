# C1 seed-row transcription correction

`seed_rows.csv` was manually transcribed from the local exploratory outputs. A post-write audit found that several values in **only the `cluster_val_acc` column** were copied incorrectly.

The scientific runner outputs, all NMSE values, recompiled validation accuracies, direct-control metrics, and `RESULT.json` aggregates were not affected.

Use `seed_rows_corrected.csv` for per-seed analysis. Retain the original `seed_rows.csv` only as provenance of the transcription mistake; do not use its `cluster_val_acc` column for inference or reporting.

Correction was made after the C1 outcomes were already known and does not alter any experimental condition, seed, model fit, threshold, or numerical runner output.
