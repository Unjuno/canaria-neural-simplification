# C71E lineage

## Parent evidence

- C68E repaired robust-teacher task validity at Gaussian sigma `.36`.
- C69E stopped because P32 was not a valid strong reference.
- C70E used all 64 calibration QR directions; full-basis calibration residual reconstruction was numerically exact, but final P64 reference validity still failed.

## C71E role

C71E localized the remaining post-target-construction loss using fresh seeds and three stages:

- H64: P64-adapted hierarchy before final compilation;
- S64: standard hierarchy-target final compiler;
- D64: same-budget direct robust-teacher-target compiler.

Terminal decision: `LOCALIZE_SHARED_MAPPING_OR_CALIBRATION_LIMIT`.

All H64/S64/D64 stage-reference safeguards failed. D64 nevertheless improved over S64 in validation accuracy, indicating measurable hierarchy/stacked-distillation loss, but the direct same-budget mapping remained invalid. Therefore the evidence does not support a unique hierarchy or compiler bottleneck.

## Next evidence class

The next valid experiment is a new exploratory two-factor mapping study on the D64 bypass path. It should separate calibration quantity from replacement capacity, e.g. nested `192` versus `384` calibration samples crossed with `4096` versus `8192` TinyRes parameters. Reduced interface selection remains closed until a valid reference mapping is obtained.

## Architecture boundary

This is C68E-repaired repository Residual-MLP evidence. It is separate from imported Residual CNN C59/C60 and SmallViT evidence.
