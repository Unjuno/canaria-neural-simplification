# C62R result — P2 frontier and basis mechanism exploration

## Status

**ADVANCE_QR_P2_TO_C63R** — prospective exploratory result only.

This experiment belongs to the C61R **Residual-MLP** recursive-hierarchy testbed. It is not a continuation or confirmation of the imported C59/C60 **Residual CNN** evidence line.

## Locked primary exploratory comparison

Fresh model seeds: 60400–60415. Eligible: 16/16. Held-out test: not used.

QR P2 versus QR P4 under additive Gaussian input noise sigma=0.04, fixed 192-sample calibration subset, paired shifted tensors, and the same nested QR residual basis:

- validation accuracy, P2 minus P4 mean: **-0.208334 percentage points**
- paired 100,000-bootstrap 95% interval: **[-0.555557, +0.115741] pp**
- preregistered exploratory non-inferiority margin: **-2.0 pp** -> advance gate PASS
- P2/P4 NMSE geometric-mean ratio: **1.013455**
- paired bootstrap 95% interval: **[0.995955, 1.030360]**
- preregistered exploratory ratio margin: **1.25** -> advance gate PASS
- informative QR-P2 delta NMSE versus frozen mean: **-0.00161961**

Therefore the locked exploratory decision is **ADVANCE_QR_P2_TO_C63R**. This does **not** establish that P2 is sufficient or minimal; a separate fresh confirmatory C63R is required.

## Mechanism observations — exploratory

The residual itself is not approximately rank-2 under ordinary unweighted squared-error geometry:

- entropy effective rank mean: **10.236**
- stable rank mean: **4.216**
- optimal SVD residual-energy fraction: k=2 **0.4103**, k=4 **0.6587**, k=8 **0.9083**

Yet P2 stayed inside the exploratory P4 non-inferiority gates. Hence a simple statement such as “the residual is intrinsically two-dimensional” is not supported.

Subspace capture differed strongly by basis:

- QR shifted-validation residual-energy capture mean: k=2 **0.2300**, k=4 **0.4590**
- SVD shifted-validation capture mean: k=2 **0.3690**, k=4 **0.6109**
- fixed-random shifted-validation capture mean: k=2 **0.0313**, k=4 **0.0627**

However, the energy-optimal SVD basis did not show a clear downstream advantage over QR after adaptation and compilation. SVD-P2/QR-P2 NMSE ratio was **0.9989** with 95% interval **[0.9834, 1.0161]**; SVD-P4/QR-P4 was **1.0046** with interval **[0.9860, 1.0248]**. This suggests that plain residual-energy capture is not identical to task-preserving correction value.

QR-P2 had a small exploratory NMSE advantage over fixed-random P2: geometric ratio **0.98765**, 95% interval **[0.97623, 0.99997]**. Validation accuracy did not separate clearly. This supports investigating task-weighted or Jacobian/Fisher-weighted geometry rather than only Euclidean residual rank.

All basis-control comparisons and Spearman correlations are mechanism-generating analyses without multiplicity-corrected confirmatory interpretation.

## Scientific boundary

Safe interpretation:

> In the exact C62R Residual-MLP Gaussian-shift testbed, QR-P2 passed prospectively locked exploratory advance gates against QR-P4, motivating a separate fresh confirmation.

Not supported:

- “P2 is confirmed sufficient.”
- “Two correction dimensions are minimal.”
- “The residual has rank two.”
- “SVD is worse/better than QR in general.”
- any claim about the imported Residual CNN C59/C60 line.
