> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Matched seen-digits regime cohort — completed once

Scientific implementation: 6d9e60a127aa83f3e966ced17767a980441593db.
Protocol/seed lock: 4e86ee9, before scientific execution.
Started 2026-10-27 00:53 UTC; terminal 00:54:04 UTC. All 16 seeds completed with exit 0; no abort, exclusion, replacement, or retry.

## Preregistered gates

- Mean paired log2(NMSE J / L): **-1.168441**, paired-bootstrap 95% CI **[-1.247279, -1.092253]**. Feature gate PASS.
- Mean paired accuracy J − L: **+1.759259 percentage points**, CI **[+1.157407, +2.337963] pp**. Preregistered noninferiority margin −2 pp: PASS.
- Conditional training-regime support criterion met. J NMSE lower in 16/16 pairs.
- Mean NMSE: L **0.146490**, J **0.065250**, architecture comparator C **0.054275**.
- Quality conjunction NMSE ≤ .08 and accuracy ≥ teacher − .02: L **0/16**, J **12/16**, C **15/16**.
- Full rows, unrounded values, medians/ranges and raw paired differences are preserved in SUMMARY.json and VERIFICATION.json.

## Evidence and checks


The deterministic report_saved.py verified every cohort/seed manifest, recomputed saved validation endpoint metrics and integer accuracy counts, confirmed 600 updates / 76,800 examples per parameter in all arms, checked schedule identity/hash, and reproduced fixed bootstrap analysis exactly. It did not reload real images or retrain models. These are implementer checks; independent saved-output audit subsequently passed 2,128 checks with zero discrepancies; its report and exact hash-bound results are included. Audit did not replay training trajectories.

Each seed scientific body took 4.45–5.11 s; process startup plus body took approximately 7–8 s. Maximum per-seed process RSS high-water was 417,152 KiB. These measurements establish this cohort fit within stated caps; no hardware benchmark claim. Arm-end RSS is cumulative process high-water, not isolated arm memory.

## Interpretation boundaries

This is new optimization randomness on the already-seen digits split, not untouched-data confirmation or independent-dataset replication. Matched initialization, minibatch schedule and update count address those nuisance differences between L and J; teacher forcing and cross-map gradient flow remain bundled treatment differences. It is not objective-only causal isolation, nor equal FLOPs/compute. C is a separate architecture and cannot rescue the primary criterion. Historical CIFAR/Issue87 results are unchanged. No test-partition arrays/outcomes were processed, publication or CI run, new charge incurred, or user-PC operation performed.
