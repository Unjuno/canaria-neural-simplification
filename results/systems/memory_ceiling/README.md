# Systems S3: constrained-memory learned-payload streaming

This directory contains the locked protocol, pre-outcome payload-identity amendment, machine-readable result, and report for Issue #23.

Read in this order:

1. `PROTOCOL_LOCK.json` — original locked S3 memory-ceiling protocol.
2. `PROTOCOL_AMENDMENT_BEFORE_CONSTRAINED_OUTCOMES.json` — identity-check-only amendment made before any constrained outcome was inspected.
3. `RESULT.json` — machine-readable PASS result.
4. `REPORT.md` — human-readable interpretation and boundaries.

The key systems result is narrowly scoped: under the locked Linux `RLIMIT_AS` configuration with +64 MiB address-space headroom, full retention of the S2 learned-payload amplification workload fails for memory while one-chunk streaming completes with the exact expected checksum.

This is not a physical-device deployment result and not scientific composition evidence.
