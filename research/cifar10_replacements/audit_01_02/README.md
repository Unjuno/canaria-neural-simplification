# Independent audit: PASS

All reported classification counts and six <=2 percentage-point flags were independently reproduced. Centered NMSE differs from original reports by at most 2.220446049250313e-16 (float64 reduction rounding). No test member was opened or decoded; no retraining or optimizer steps were performed.

| Seed | Teacher correct | Arm | Correct | Drop (pp) | Centered NMSE | <=2pp criterion |
|---|---:|---|---:|---:|---:|---|
| 26102601 | 4550/5000 | Comp-joint | 4365/5000 | 3.70 | 0.281001392416 | Fail |
| 26102601 | 4550/5000 | Sep-joint | 4292/5000 | 5.16 | 0.303173831135 | Fail |
| 26102601 | 4550/5000 | Sep-local | 4109/5000 | 8.82 | 0.384967690338 | Fail |
| 26102602 | 4559/5000 | Comp-joint | 4382/5000 | 3.54 | 0.242675324259 | Fail |
| 26102602 | 4559/5000 | Sep-joint | 4314/5000 | 4.90 | 0.276284739280 | Fail |
| 26102602 | 4559/5000 | Sep-local | 4247/5000 | 6.24 | 0.309627048880 | Fail |

## Verified

- Source/approved-plan/protocol Git chronology and SHA256 bindings; all checkpoint envelope and artifact hashes; unchanged original evidence bytes.
- Fixed-final teachers epoch120, exact identities and completed quality gates.
- Reconstructed PCG64 teachertrain/calibration/validation partitions, all saved split hashes, training-only archive member hashes and teachertrain-only normalization.
- Exact seeded initialization, bitwise-identical Sep-local/Sep-joint starts with independent storage, and5120 convolution weights per arm.
- Common deterministic schedule:1000 updates,25 complete5000-image passes,125000 presentations per arm, final8 retained each pass.
- Saved final cursors1000; every Adam state counter1000, exact optimizer settings, finite parameters/moments; last and final states identical.
- Original teacher-state hash unchanged during independent functional evaluation; pinned source freezes teacher/BN/prefix/suffix and excludes them from replacement optimizers.

## Limits

Historical trajectory was not replayed. There is no retained per-update ID/gradient history or final fit-time teacher snapshot, so source plus final checkpoint counters cannot prove every historical step or teacher-state immutability. See SOURCE_REVIEW.md. All6 utility outcomes fail; this is a descriptive pilot criterion, not an inference about a population. Third planned seed remains outstanding. Official test remains sealed.

## Files

- `audit.py`: standalone successful verifier, no production experiment imports.
- `audit_result.json`, `result_seed_*.json`: complete machine-readable checks and results.
- `predictions_seed_*.npz`: fixed validation IDs, labels, independently computed predictions and logits.
- `endpoint_statistics_seed_*.npz`: float64 per-feature full-sample sums and squared sums.
- `evidence_sha256.json`: all original evidence files read, verified unchanged after audit.
- `PLAN.md`, `SOURCE_REVIEW.md`, `stdout.log`: scope, source review and successful process record.
- `audit_attempt1.py`, `attempt1.stdout.log`: preserved pre-data audit implementation failure (tuple versus JSON-list equality), corrected by canonical serialization without changing experimental evidence.

Successful audit process exited0; CPU4/inter-op1, deterministic PyTorch2.10.0+cpu. All work confined to this separate audit directory.
