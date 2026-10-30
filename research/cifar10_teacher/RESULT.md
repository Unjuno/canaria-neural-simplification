# Stage A: three fixed-final CIFAR-10 teachers

All three authorized seeds completed exactly 120 epochs and exited 0. The
prelaunch protocol was committed as `cfcc08223ac9aef556384581c6c92b04a75e5714`
from reviewed source `67bd076ec10574dc372b86f94e1d937c8c324189`. Every frozen
source file still matches its prelaunch SHA256.

| Seed | Final validation correct / 5000 | Accuracy | >=85% gate | Elapsed seconds |
|---|---:|---:|---|---:|
| 26102601 | 4550 | 91.00% | PASS | 8107.587 |
| 26102602 | 4559 | 91.18% | PASS | 8348.296 |
| 26102603 | 4545 | 90.90% | PASS | 8778.655 |

Each run contains 120 epoch-completion events, implying 37,560 updates under
the frozen 313-batch epoch schedule. No early validation, best-checkpoint
selection, tuning, seed substitution, restart or cap extension occurred.
Calibration was neither trained nor evaluated; official test was never opened
or deserialized. Full-archive checksum/traversal necessarily covers compressed
test bytes. All durations were below the original ten-hour per-seed caps.

Seeds 01 and 02 ran concurrently, each with four intra-op/one inter-op threads.
Seed03 launched only after seed01's observed exit0, following another available
RAM check (8,366,352 KiB). The maximum simultaneous teacher workers was two.
After seed02 finished, other separately authorized work could use the free
four-thread slot; this report does not claim isolated hardware timing.

`evidence/` preserves immutable launch records, full epoch logs, stdout/stderr,
observer progress history, terminal exit receipts, final optimizer/RNG-bearing
checkpoints, evaluator exports, RAM checks and the complete SHA256 manifest.
The duplicate latest checkpoints remain in local evidence and their hashes
are in terminal receipts. Public dataset/cache files are excluded.

Trainer validation retained counts, not raw per-example predictions. Independent
admission/recomputation is a separate audit, not represented as performed by
this training owner. Initial worker OS PIDs were unavailable across isolated
exec PID namespaces; authoritative tool session handles are recorded. No CI,
network downloads, push, merge, fees or user-PC execution were used by this run.
