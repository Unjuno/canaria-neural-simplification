# C71E result — P64 pipeline bottleneck localization

## Status

**PROSPECTIVE EXPLORATORY**. Protocol was locked before outcomes at commit `56258365a19e96880e49e37b6e17ee21ce683d44`. Fresh seeds were `69400–69415`; all 16 were eligible. Held-out test data were not used.

## Terminal decision

`LOCALIZE_SHARED_MAPPING_OR_CALIBRATION_LIMIT`

The repaired target remained task-valid, but none of the three locked downstream stages met the `-5 pp` task-reference safeguard.

| stage | meaning | mean stage−teacher (pp) | bootstrap95 (pp) | validity |
|---|---|---:|---:|---|
| H64 | P64-adapted hierarchy before final compilation | -5.9491 | [-7.0139, -4.8611] | FAIL |
| S64 | standard compiler fitted to H64 targets | -5.4630 | [-6.6435, -4.3750] | FAIL |
| D64 | same-budget compiler fitted directly to robust-teacher targets | -4.6991 | [-5.6713, -3.7731] | FAIL |

Target validity:

- robust clean accuracy mean: `97.5231%`
- robust shifted accuracy mean: `85.7407%`
- clean→shift mean: `-11.7824 pp`
- bootstrap95: `[-12.8472,-10.7639] pp` → PASS vs `-20 pp`

P64 full-basis implementation remained numerically exact on calibration residuals; maximum relative reconstruction squared error was below the locked `1e-10` invariant.

## Descriptive stage diagnostics

- S64−H64 validation accuracy: `+0.4861 pp`, bootstrap95 `[-0.1620,+1.1806] pp`
- D64−S64 validation accuracy: `+0.7639 pp`, bootstrap95 `[+0.1157,+1.3889] pp`
- H64 activation NMSE vs robust teacher mean: `0.09518`
- S64 activation NMSE vs robust teacher mean: `0.09967`
- D64 activation NMSE vs robust teacher mean: `0.08218`
- S64 NMSE vs H64 mean: `0.00750`
- H64 calibration NMSE vs robust teacher mean: `0.00214`
- D64 calibration NMSE vs robust teacher mean: `0.00767`

D64's positive accuracy improvement over S64 shows that bypassing hierarchy-target distillation recovers some task performance. However D64 still fails the locked task-reference safeguard, so the experiment cannot localize the remaining loss solely to hierarchy adaptation.

## Scientific interpretation

C70E already ruled out missing calibration-basis dimension as a sufficient explanation. C71E further shows:

1. **H64 fails before final compilation**, so standard compilation is not the sole origin of the gap.
2. **D64 improves on S64**, so hierarchy/stacked-distillation contributes measurable loss.
3. **D64 still fails**, so bypassing the hierarchy is insufficient under the current `192` calibration samples and `4096`-parameter mapping budget.

The remaining uncertainty is therefore a shared limit involving at least one of:

- 4096-parameter function capacity;
- 192-sample calibration/generalization;
- optimization of the direct mapping;
- interactions among these factors.

C71E does **not** distinguish those mechanisms.

## Safe statement

> In the C68E-repaired Residual-MLP target at Gaussian sigma `.36`, C71E found that P64 hierarchy, standard compiled replacement, and same-budget direct teacher-target compiler all missed the preregistered task-reference safeguard. Direct teacher fitting improved over standard hierarchy-target compilation, but did not restore validity. The unresolved bottleneck is therefore shared by the fixed mapping/calibration regime rather than being attributable solely to missing residual-basis dimension or final compilation.

## Not supported

- the 4096-parameter budget alone is proven insufficient;
- 192 calibration samples alone are proven insufficient;
- hierarchy adaptation is irrelevant;
- the final compiler is irrelevant;
- P8/P16 is a valid frontier;
- any claim about imported Residual CNN C59/C60.

## Next gate

Use a prospectively locked factorial-style experiment on the **D64 direct teacher-target path** to separate calibration quantity from mapping capacity. The cleanest next design compares at least the current `(192 samples, 4096 params)` baseline against a larger-calibration condition at the same capacity and a larger-capacity condition at the same calibration size, with a combined larger-calibration/larger-capacity cell for interaction diagnostics.
