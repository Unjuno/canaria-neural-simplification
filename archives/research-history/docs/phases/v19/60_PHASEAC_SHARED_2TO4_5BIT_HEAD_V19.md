# Phase AC — Shared-pattern 2:4 head with 5-bit values (exploratory)

Built on Phase-AB 296 B Conv3-q4 core. To reduce 2:4 index overhead, groups of output rows share the same 2-of-4 input pattern. Saved index bits are spent on 5-bit retained head weights while keeping whole-model storage below 10 KB.

Conditions: output-pattern sharing group size 4, 8, 48; retained first-head weights signed calibrated 5-bit with one FP16 scale per output row; first-head bias FP16; other shell channelwise 4-bit/FP16 bias; core 296 B. No coefficient refit and no extra repair beyond tau=8 shell repair.

Initial cohort: seeds 3000-3007, exploratory.