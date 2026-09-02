# C63R result — QR-P2 versus QR-P4 fresh confirmation

## Status

**C63R_CONFIRMATORY_PASS**

C63R is a prospectively locked fresh confirmation selected from the C62R exploratory advance. It belongs to the C61R/C62R/C63R **Residual-MLP** recursive-hierarchy testbed and is not evidence about the imported C59/C60 **Residual CNN** line.

## Locked confirmatory result

Fresh model seeds: 61400–61415. Eligible: 16/16. Missing rows: 0. Held-out test: not used.

Under additive Gaussian input noise sigma=0.04, fixed 192-sample calibration subset, and a nested residual-QR interface:

- validation accuracy, P2 minus P4 mean: **-0.370371 percentage points**
- paired 100,000-bootstrap 95% interval: **[-0.601853, -0.162037] pp**
- preregistered non-inferiority margin: **-2.0 pp** -> PASS
- P2/P4 NMSE geometric-mean ratio: **1.009512**
- paired bootstrap 95% interval: **[0.996028, 1.022772]**
- preregistered NMSE-ratio margin: **1.25** -> PASS
- informative P2 delta NMSE versus frozen mean: **-0.000240125**

Both primary gates passed, therefore the locked decision is **C63R_CONFIRMATORY_PASS**.

## Interpretation

The validation interval is entirely below zero. Therefore P2 was slightly worse than P4 in validation accuracy in this fresh cohort. The result is **non-inferiority**, not equality and not superiority.

Safe statement:

> In the exact C63R Residual-MLP recursive-hierarchy sklearn-digits testbed under additive Gaussian sigma=0.04, fixed 192-sample calibration, and the locked nested-QR correction procedure, P2 was non-inferior to P4 under the preregistered validation-accuracy and NMSE-ratio margins.

Not supported:

- P2 equals P4.
- P2 is better than P4.
- P2 is the minimum possible correction dimension.
- 2/32 is a universal interface requirement.
- any statement that C63R confirms the imported Residual CNN C59/C60 line.

Establishing a lower frontier would require a separately prospective P1/P0 experiment; the present result alone cannot establish minimality.
