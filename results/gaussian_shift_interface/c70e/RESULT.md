# C70E result — P64 full-basis reference repair

## Status

**PROSPECTIVE EXPLORATORY**. Protocol was locked before outcomes at commit `d1df83b8a242efbff9a353c9c1882843a4606d3d`. Fresh seeds were `68400–68415`; all 16 were eligible. Held-out test data were not used.

## Terminal decision

`STOP_P64_REFERENCE_INVALID`

P64 used all 64 columns of the canonical calibration QR basis. The implementation invariant confirmed that this full basis reconstructed the shifted-calibration teacher residual to numerical precision:

- maximum relative calibration residual squared error across eligible seeds: `2.3482e-13`
- mean: `1.5961e-13`
- locked threshold: `1e-10` → implementation invariant PASS

This is implementation equivalence on calibration residuals, not a scientific reference-validity result.

## Scientific gates

Robust target validity remained PASS:

- robust clean accuracy mean: `97.5694%`
- robust shifted accuracy mean: `85.2315%`
- shifted-minus-clean mean: `-12.3380 pp`
- 95% CI: `[-13.5417,-11.1111] pp`
- margin: `-20 pp`

P64 reference validity failed narrowly:

- P64 shifted accuracy mean: `81.3889%`
- robust shifted teacher mean: `85.2315%`
- P64-minus-teacher mean: `-3.8426 pp`
- 95% CI: `[-5.1157,-2.5926] pp`
- locked lower-bound margin: `-5 pp` → **FAIL**

## Descriptive P64 vs P32

P64 did improve the weaker P32 reference:

- P64−P32 validation accuracy mean: `+0.7176 pp`
- 95% CI: `[+0.2546,+1.1574] pp`
- P64/P32 NMSE geomean: `0.9863`
- 95% CI: `[0.9718,1.0002]`

These were descriptive diagnostics, not additional gates.

## Interpretation

The experiment exhausts missing calibration-basis dimension as the simple explanation for C69E's P32 failure. P64 reconstructs the calibration residual in the full 64-dimensional hidden space, yet the final adapted/compiled replacement still fails the task-reference safeguard. Therefore the remaining loss must arise **after** full-basis calibration target construction: from generalization of top-boundary adaptation, final compilation, or their interaction.

Safe statement:

> In the C68E-repaired Residual-MLP target at Gaussian sigma `.36`, C70E showed that full 64-column calibration-basis residual correction was numerically exact on the calibration residuals but still did not yield a valid final reference after the locked adaptation/compilation pipeline. Basis dimension alone therefore does not explain the reference gap in this protocol.

Not supported:

- P64 is universally insufficient;
- the failure is already localized specifically to the final compiler rather than top-boundary adaptation;
- P8/P16 from C69E are valid frontier candidates;
- any claim about imported Residual CNN C59/C60.

## Next gate

A fresh stage-localization experiment should separately evaluate:

1. the P64-adapted hierarchy **before** final compilation;
2. the standard compiled P64 final replacement;
3. a same-budget `TinyRes(64,32)` fitted **directly to repaired-teacher calibration activations**, bypassing hierarchy-target distillation.

This distinguishes hierarchy/top-boundary failure from final-compiler capacity or stacked-distillation loss.
