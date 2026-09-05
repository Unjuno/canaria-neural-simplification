# C69E result — robust-teacher interface frontier with P32 reference

## Status

**PROSPECTIVE EXPLORATORY**. Protocol was locked before outcomes at commit `1f23889de0050d7b87d0579e730507f4d72ed7f8`. Fresh model seeds were `67400–67415`; all 16 were eligible. Held-out test data were not used.

## Terminal decision

`STOP_P32_REFERENCE_INVALID`

The C68E repaired teacher reproduced task validity on the fresh C69E cohort, but the prospectively designated P32 strong reference failed its own reference-validity safeguard. Candidate dimensions therefore cannot be selected from this experiment even though P8 and P16 passed their numerical candidate gates against P32.

### Validity gates

- robust-teacher clean validation accuracy mean: `97.2917%`
- robust-teacher shifted validation accuracy mean: `85.8333%`
- robust shifted-minus-clean mean: `-11.4583 pp`
- bootstrap95: `[-12.6157, -10.3241] pp`
- target validity margin: `-20 pp` → **PASS**

P32 reference:

- P32 shifted-validation accuracy mean: `80.3241%`
- robust shifted-teacher accuracy mean: `85.8333%`
- P32-minus-teacher mean: `-5.5093 pp`
- bootstrap95: `[-6.5046, -4.5602] pp`
- reference validity margin: `-5 pp` → **FAIL**

The failure criterion uses the bootstrap lower bound; the lower bound is below `-5 pp`.

## Candidate curve — descriptive only

Because P32 failed the locked reference-validity gate, the following candidate comparisons do **not** support candidate selection.

| candidate | val diff vs P32 mean (pp) | val 95% CI (pp) | NMSE ratio vs P32 | NMSE 95% CI | joint |
|---|---:|---:|---:|---:|---|
| P0 | -1.1343 | [-2.1296, -0.2083] | 1.4736 | [1.4085, 1.5383] | FAIL |
| P1 | -0.5324 | [-1.3426, +0.3009] | 1.4169 | [1.3626, 1.4689] | FAIL |
| P2 | -0.6944 | [-1.6898, +0.3241] | 1.3699 | [1.3214, 1.4168] | FAIL |
| P4 | -0.6944 | [-1.3426, +0.0926] | 1.2770 | [1.2428, 1.3085] | FAIL |
| P8 | -0.1389 | [-0.8333, +0.6019] | 1.1396 | [1.1147, 1.1618] | PASS |
| P16 | +0.3472 | [-0.1389, +0.8565] | 1.0339 | [1.0182, 1.0491] | PASS |

Locked candidate margins were `-2 pp` for validation non-inferiority and `1.25` for the candidate/P32 NMSE ratio. The observed pass pattern was `[false,false,false,false,true,true]`, but **no dimension is selected** because reference validity has precedence.

## Scientific interpretation

The robust teacher is a valid target at sigma `.36`, but P32 is not close enough to that target under the locked `-5 pp` accuracy safeguard to serve as the reference for an interface frontier. This shifts the next question from reduced-interface selection to **reference construction**.

P32 was never claimed to be mathematically full residual reconstruction. The correct next experiment is therefore to test a full 64-column QR calibration-basis correction (P64) as a stronger reference on fresh seeds. If P64 still fails, the bottleneck is more likely top-boundary adaptation / compilation capacity than missing residual-basis dimension.

## Safe statement

> In the C68E-repaired Residual-MLP target at Gaussian sigma `.36`, C69E confirmed that the robust target remained task-valid, but the prospectively designated P32 strong reference failed the locked reference-validity safeguard. Therefore no reduced interface dimension was selected from C69E.

## Not supported

- P8 is the robust-teacher interface frontier;
- P8 or P16 is sufficient relative to the repaired teacher itself;
- P32 is exact/full reconstruction;
- robust training universally raises the required interface dimension;
- any claim about the imported Residual CNN C59/C60 line.
