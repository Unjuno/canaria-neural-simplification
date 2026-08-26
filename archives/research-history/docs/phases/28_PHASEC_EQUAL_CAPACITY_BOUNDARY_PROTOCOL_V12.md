# Phase C Equal-Capacity Boundary Adapter Protocol v12

Status: FROZEN BEFORE OUTCOMES.
Frozen: 2026-08-23 JST.

## Goal
Separate repair *location* from repair *capacity* after aggressive full-span compilation.

## H
H1 (pure post-location effect): with exactly the same adapter architecture, initialization, trainable parameter count, optimizer, data order, and repair budget, an adapter placed immediately after the compiled span recovers more matched-control utility than the same adapter immediately before the compiled span.
H2 (capacity-response): recovery increases with adapter bottleneck rank r.

## T
- Data/model: sklearn digits; residual 8-block CNN, ch=8, residual scale=0.5.
- New model seeds: 1200..1207, independent of prior 1000..1107.
- Baseline: AdamW lr=3e-3, wd=1e-4, 24 epochs, batch=256.
- Eligibility: clean validation accuracy >=0.95 before intervention.
- Full-span intervention: fit one frozen Conv1 replacing blocks 0..7 using first 192 training samples, identical to Phase B.
- Symmetric boundary adapter: residual pointwise MLP, x + W_up(ReLU(W_down(x))). Both locations see 8-channel 8x8 tensors. W_up is zero initialized so both interventions start functionally identical to the no-adapter compiled model.
- Adapter ranks r={4,16,64,256}; exact trainable parameters c(r)=17r+8 = {76,280,1096,4360}.
- Locations: PRE immediately before compiled Conv1; POST immediately after compiled Conv1.
- Only adapter parameters are trainable; compiled Conv1 and all original shell parameters remain frozen.
- Repair budgets tau={0,1,2,4,8}; AdamW lr=7e-4, wd=1e-4.
- Matched no-compile control receives the same augmented continuation.
- Primary utility U = augmented accuracy(intervention)/augmented accuracy(matched control).
- Utility floor =0.95.

## D
Primary location test at n=8 eligible seeds, r=256, tau=8:
- PASS pure post-location if lower 95% paired seed-bootstrap CI of mean(U_POST-U_PRE) > 0.05.
- FAIL pure post-location if upper 95% CI <=0.05.
- Otherwise UNCERTAIN.

Secondary:
- Report paired POST-PRE differences for every r,tau without threshold changes.
- Capacity-response is descriptive unless all four seed-paired mean utilities are monotone nondecreasing at tau=8.
- Report PASS95 rates, but do not redefine the primary endpoint based on them.

## C
Alternatives: Conv1 intervention may create asymmetric information geometry; pointwise adapters may be too weak; zero-init optimization may delay PRE/POST differently; distributed head adaptation may require nonlocal parameters rather than boundary-local capacity.

## U
Use 10,000 percentile bootstrap resamples over the eight paired model seeds. The experimental unit is seed, not individual examples. This is within-task/architecture evidence only.
