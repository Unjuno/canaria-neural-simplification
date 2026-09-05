# C68E lineage

## Parent evidence

- C65R: P0 versus P2 confirmatory non-inferiority PASS at sigma=.04 for the clean-trained Residual-MLP teacher.
- C66R: no exploratory P0 failure through sigma=.20.
- C67E: teacher task validity, not P0-versus-P2, failed first at sigma=.36; terminal decision `STOP_VALIDITY_BOUNDARY_AT_SIGMA_0_36`.

## C68E role

C68E did **not** test interface dimension. It isolated the target-validity problem identified by C67E and evaluated one prospectively locked paired teacher-training repair at sigma=.36.

Terminal decision: `ADVANCE_REPAIRED_TEACHER_TO_C69E`.

All three repair gates passed on fresh seeds. Therefore the augmented teacher recipe may be used as an exploratory target construction in a separately locked C69E interface experiment.

## Non-transfer boundary

The repaired teacher has different learned parameters and potentially different boundary geometry from the clean-trained teacher. C65R/C66R P0 conclusions therefore cannot be carried forward by assumption. C69E must re-measure the interface frontier with fresh seeds and a stronger reference correction.

This line remains repository Residual-MLP evidence, separate from imported Residual CNN C59/C60 and SmallViT evidence.
