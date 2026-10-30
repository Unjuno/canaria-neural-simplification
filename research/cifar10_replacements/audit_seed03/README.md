# All three replacement seeds independently verified

Seed03 independent audit PASS, exit0. Counts: teacher4545/5000, Comp4352, Sep-joint4240, Sep-local4223. Drops:3.86,6.10,6.44 pp; all fail the descriptive <=2pp criterion. Separate functional inference reproduces centered NMSE within2.78e-16. Hash/schedule/initialization/5120-parameter/1000-step optimizer checks passed. Official test was not opened or decoded. No training or optimizer steps occurred.

The prior01/02 audit is preserved unchanged at `/workspace/shared/canaria_cifar_replacement_audit_20261026`. Its manifest and every listed output/input hash were rechecked. `cohort_audit_manifest.json` binds all three independently recomputed results to the original teacher, replacement, initialization, recipe, validation, approved-plan and prediction hashes. This is an evidence manifest, NOT official-test release authorization.

## Entire intended cohort (correct counts /5000)

| Seed | Teacher | Arm | Correct | Drop (pp) | Centered NMSE | <=2pp |
|---|---:|---|---:|---:|---:|---|
| 26102601 | 4550 | Comp-joint | 4365 | 3.70 | 0.281001392416 | Fail |
| 26102601 | 4550 | Sep-joint | 4292 | 5.16 | 0.303173831135 | Fail |
| 26102601 | 4550 | Sep-local | 4109 | 8.82 | 0.384967690338 | Fail |
| 26102602 | 4559 | Comp-joint | 4382 | 3.54 | 0.242675324259 | Fail |
| 26102602 | 4559 | Sep-joint | 4314 | 4.90 | 0.276284739280 | Fail |
| 26102602 | 4559 | Sep-local | 4247 | 6.24 | 0.309627048880 | Fail |
| 26102603 | 4545 | Comp-joint | 4352 | 3.86 | 0.258026136009 | Fail |
| 26102603 | 4545 | Sep-joint | 4240 | 6.10 | 0.291763504419 | Fail |
| 26102603 | 4545 | Sep-local | 4223 | 6.44 | 0.301117443387 | Fail |

All nine outcomes fail <=2pp. This is a descriptive three-seed pilot, not statistical noninferiority or a population-generalization result. Comp/Sep architecture, receptive field and depth differ; only Sep-local/Sep-joint share architecture and exact initialization.

## Evidence and limits

`audit.py` imports no production experiment code. It independently reconstructs the allowed train-only data split/normalization, seeded initializations and schedules, then uses a separate functional forward and NumPy float64 aggregate statistics. Saved NPZs retain IDs/labels/predictions/logits and per-feature teacher statistics. All original evidence bytes remained unchanged.

Teacher parameters/BN/suffix are frozen by the pinned source and excluded from saved replacement optimizers; independent inference uses unchanged original teacher state. Historical every-step behavior was not replayed and no final fit-time teacher snapshot exists, so the final counters/source cannot prove every historical update or historical immutability by themselves. See the full manifest limitations and the preserved prior SOURCE_REVIEW.md.

The4-thread numerical audit process exited0 and the slot was explicitly released at14:17 UTC. Cohort assembly afterward used standard library only. No network/CI or original evidence writes.
