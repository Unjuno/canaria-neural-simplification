# Phase D Equal-Capacity Spatial Boundary Adapter Protocol v12

Status: FROZEN BEFORE PHASE-D OUTCOMES.

## Goal
Test whether Phase-C failure of a pure POST location effect was caused by a too-weak pointwise adapter class. Use identical spatial adapters before vs after the compiled span, reaching capacity comparable to the full classifier head.

## H
H1 spatial post-location: with identical spatial adapter architecture/capacity and training, POST recovers more utility than PRE.
H2 local-capacity sufficiency: a ~20k-parameter boundary-local spatial adapter can recover U>=0.95 after aggressive Conv1 compilation.

## T
- Independent seeds 1300..1307.
- Same digits residual-8 baseline/training/eligibility/full-span Conv1 intervention as Phase C.
- Residual spatial adapter: x + Conv1_up(ReLU(Conv3_down(x))). Up projection zero initialized.
- ranks r={8,64,256}; exact trainable params c(r)=81r+8={656,5192,20744}.
- PRE vs POST placements are symmetric and see identical 8-channel 8x8 tensors.
- Only adapter trainable; original shell and compiled Conv1 frozen.
- tau={0,1,2,4,8}; same repair data, optimizer and PRE/POST data-order stream.
- Matched no-compile continuation control.

## D
Primary at r=256,tau=2 (chosen prospectively because prior independent Phase A/B localized the adaptive onset near tau=2):
- pure POST PASS if lower 95% paired seed-bootstrap CI of mean(U_POST-U_PRE)>0.05;
- FAIL if upper CI<=0.05; else UNCERTAIN.
- local-capacity sufficiency PASS for a location if lower 95% CI of its mean U is >=0.95; FAIL if upper CI<0.95; else UNCERTAIN.
Secondary report tau=8 and lower ranks without threshold changes.

## C
If neither location recovers at ~20k params, compensation likely requires distributed/nonlocal shell degrees of freedom rather than boundary-local capacity. If both recover similarly, capacity/function class matters but location does not.

## U
Experimental unit is seed; n=8. 10,000 percentile paired bootstrap resamples.
