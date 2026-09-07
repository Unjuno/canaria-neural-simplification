# QR prefix/order dependence: implementation diagnostic

Evidence class: **POST_HOC_MECHANISM_DIAGNOSTIC**, not a fresh confirmation. This is an overlooked implementation property, not a claim of a new linear-algebra theorem. It does not invalidate the original fixed-protocol performance measurements.

## Source and observation

`canonical_nested_qr` in `scripts/gaussian_shift_interface/run_c61r_seed.py` computes unpivoted `torch.linalg.qr(residual.T, mode="reduced")`. The correction takes the first k columns. Canonicalizing signs removes sign ambiguity; it does not make the subspace sample-order invariant or select the highest-energy residual directions.

When the first k residual rows are linearly independent, this projector is determined by those first k calibration residual rows. Appending further rows without changing that prefix does not change the projector in exact arithmetic. Training the correction mapping still uses all calibration samples: this statement concerns basis construction, not the whole training procedure.

## Variables and assumptions

All quantities below are dimensionless (SI unit 1); hidden activations use model coordinates, not a physical unit.

| Symbol | Meaning | SI unit | Definition / shape | Domain / assumptions | Type |
|---|---|---|---|---|---|
| n | 校正標本数 | 1 | residual row count | integer, n >= d | scalar integer |
| d | 隠れ状態次元 | 1 | residual column count | positive integer; implementation d=64 | scalar integer |
| k | 射影次元 | 1 | number of retained QR columns | integer, 1 <= k <= d; first k residual rows independent | scalar integer |
| R | 校正残差 | 1 | n by d matrix | real finite entries | matrix |
| A | QRへの入力 | 1 | R transpose, d by n | first k columns independent | matrix |
| A_k | QR入力の先頭k列 | 1 | A[:, :k], d by k | rank k | matrix |
| Q | QR直交因子 | 1 | d by d | Q transpose times Q = identity | matrix |
| T | QR上三角因子 | 1 | d by n | unpivoted factorization A=QT | matrix |
| Q_k | Qの先頭k列 | 1 | d by k | orthonormal columns | matrix |
| T_kk | Tの先頭主ブロック | 1 | k by k | nonsingular under rank assumption | matrix |
| P_k | 直交射影 | 1 | Q_k times Q_k transpose, d by d | symmetric and idempotent | matrix |
| R_v | 評価標本の残差 | 1 | arbitrary number of rows by d | nonzero for energy ratio | matrix |
| c_k | 評価残差のエネルギー捕捉率 | 1 | squared projected Frobenius norm / squared total norm | [0,1] in exact arithmetic | scalar |

## Complete derivation

1. The unpivoted factorization is

   A = R^T = Q T.

2. Since T is upper triangular in its first d columns, every entry below row k in its first k columns is zero. Therefore restricting the product to its first k columns gives

   A_k = Q_k T_kk.

3. A_k has rank k by assumption, while Q_k has k orthonormal columns and thus rank k. If T_kk were singular, the product Q_k T_kk would have rank less than k, contradicting rank(A_k)=k. Hence T_kk is invertible.

4. From A_k = Q_k T_kk each column of A_k lies in the column space of Q_k. Conversely Q_k = A_k T_kk^{-1}, so every column of Q_k lies in the column space of A_k. The two column spaces are identical.

5. P_k = Q_k Q_k^T is the unique Euclidean orthogonal projector onto that column space: it acts as identity on the span of Q_k and as zero on its orthogonal complement. Consequently any other unpivoted QR factorization with the same first k independent input columns has the same P_k, irrespective of appended input columns and sign choices.

6. Reordering calibration rows can change those first k columns of A and their span; it can therefore change P_k. Full k=d is different: a square orthogonal Q gives P_d=I, so changing the basis cannot change the full-dimensional projector.

7. The diagnostic energy fraction is

   c_k = ||R_v Q_k||_F^2 / ||R_v||_F^2.

   Orthogonal projection cannot increase Euclidean norm, so this ratio is in [0,1] in exact arithmetic. It is **not** a classifier-accuracy metric or a Fisher-weighted explained fraction.

### Dimension / unit check

A_k and Q_k T_kk are both d by k. P_k is d by d, and R_v P_k has the same shape and units as R_v. The squared-norm ratio is dimensionless. No physical memory or runtime statement follows.

### Small exact example

Take n=3, d=2, k=1 and residual rows (1,0), (0,1), (1,1). The original prefix gives P_1=diag(1,0). Swapping the first two rows gives P_1=diag(0,1). For evaluation residual (1,0), capture changes from 1 to 0 although the calibration set is unchanged. At k=2 both projectors are identity.

## Executed numerical diagnostic

Three **existing verification seeds**: 69300, 70300, 71300. Rebuilt robust Residual-MLP / Gaussian .36 residuals with the same data and training recipes as the C72/C73 source. Used four fixed orders: identity, reverse, and two seeded permutations. Dimensions 1/2/4/8/16/32/64.

Local environment: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0, one torch thread. This differs from GitHub's pinned environment and these results are **not pooled into C73E or C74E**.

| Diagnostic | Observed maximum |
|---|---:|
| Projector difference after appending N192 to N384 | 0.0 |
| Projector difference from using only first k residual rows | 1.8725898e-6 |
| Float64 SVD-projector difference after reversing sample order | 3.6799345e-14 |

P2 validation residual-energy capture across the four orders:

| Known verification seed | Minimum | Maximum |
|---|---:|---:|
| 69300 | 0.121512 | 0.202606 |
| 70300 | 0.175359 | 0.216178 |
| 71300 | 0.145934 | 0.281726 |

The numerical checks also retained full-P64 projector invariance. Original residual NPZ arrays, per-dimension/per-order JSON, the original executed script, and their hashes are preserved in the session handoff artifact. The repository runner can regenerate them with `CANARIA_REPO` set to the repository directory.

## Consequences and limits

- Do not describe this QR construction as selecting the globally most important k directions from all calibration samples.
- A nested N192-to-N384 extension keeps the old prefix; it need not change a low-dimensional QR basis at all. The downstream fitting data still changes.
- P64 experiments C73E/C74E do not lose full-dimensional coverage due to this order effect.
- Next low-dimensional studies should include prospectively fixed ordering/permutation controls or a separately specified all-sample subspace construction. Such a change is a new protocol, not a silent correction of old experiments.
- No classification-performance advantage of SVD or ordering strategy is established by this construction audit.
- Near-rank-deficient prefixes require an additional numerical-conditioning check; the exact theorem assumes independence.

## ERROR CHECK

The rank assumption, sign-versus-subspace distinction, matrix shapes, SI units, full-basis control, SVD ordering control, and diagnostic-versus-task-outcome boundary have been checked. This is not independent scientific peer review and not a new confirmatory cohort.
