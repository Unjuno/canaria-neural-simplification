# C72E result and evidence closure

Evidence class: PROSPECTIVE_EXPLORATORY. Original protocol and numerical outcomes are unchanged.

Decision: `LOCALIZE_CALIBRATION_QUANTITY_REPAIR`.

| Cell | Accuracy gap to teacher (pp) | Paired 95% CI (pp) | Reference gate |
|---|---:|---|---|
| N192_W32 | -5.856482 | [-7.29166567325592, -4.606481269001961] | NOT_ESTABLISHED |
| N192_W64 | -4.976853 | [-6.319446489214897, -3.7500008940696716] | NOT_ESTABLISHED |
| N384_W32 | -3.379631 | [-4.467593505978584, -2.314816415309906] | PASS |
| N384_W64 | -2.407408 | [-3.171297535300255, -1.6435198485851288] | PASS |

Increasing the nested calibration set from 192 to 384 restored the reference safeguard in the tested direct-mapping regime. Width increase alone did not establish that safeguard. This identifies a sufficient tested intervention, not a unique cause, universal sample threshold, or full recursive pipeline repair.

A failed non-inferiority gate does not prove inferiority. Calibration and validation NMSE use different target variance denominators; a raw difference between them is not an established generalization-gap measure.

All 16 original seed records, aggregate bytes, artifact identifiers and hashes are preserved here. CLOSURE_AUDIT.json documents recomputation, not independent scientific review.
