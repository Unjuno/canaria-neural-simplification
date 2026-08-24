# G6b rollout-horizon diagnostic — v23

Exploratory mechanistic follow-up on the already-confirmed seeds 3500-3507. This does **not** change the negative confirmatory decision and is not independent evidence.

Question: how quickly does autoregressive trajectory disagreement grow despite near-preserved teacher-forced PPL?

Fixed horizons: **1, 2, 4, 8, 16, 24 generated characters**.

For each confirmatory seed, retrain the exact frozen v23 teacher/compiled condition and compute greedy compiled-vs-reference token agreement on the same 32 held-out prompts at:
- tau=0: compiled vs baseline;
- tau=8: jointly repaired compiled model vs matched continued-training control.

A single 24-character rollout is generated per model pair and shorter-horizon scores are prefixes of that same rollout, preventing horizon-specific resampling.

Report mean agreement across seeds and seed-bootstrap 95% CIs. Because this is a post-confirmatory diagnostic, no PASS/FAIL threshold is attached.
