# C75E: output-observable subspace, not a universal compression theorem

This is a standard linear-algebra derivation applied to the repository's fixed affine classifier head. The experimental question is whether a dimension chosen from this observable structure remains adequate AFTER hierarchy adaptation and compilation. The identities themselves are not a novel theorem.

## Variables / units

All entries use SI unit 1 (dimensionless model coordinates, not physical memory/energy units).

| Symbol | Meaning (Japanese) | SI | Definition / shape | Domain / assumption | Type |
|---|---|---|---|---|---|
| d,m | 隠れ状態次元、クラス数 | 1 | d=64,m=10 here | positive integers, m>=2 | scalar integers |
| W,b | 固定分類ヘッドの重み、バイアス | 1 | m by d; m-vector | finite real entries | matrix, vector |
| h_T,h_0 | 教師、基準階層の隠れ状態 | 1 | d-vectors | arbitrary finite states | vectors |
| r | 補正残差 | 1 | h_T-h_0 | d-vector | vector |
| I_m,I_d | 単位行列 | 1 | m by m,d by d | constants | matrices |
| 1_m | 全成分1 | 1 | m-vector | constant | vector |
| C | クラス平均除去射影 | 1 | I_m-1_m 1_m^T/m | symmetric, idempotent | matrix |
| W_c | 中心化ヘッド | 1 | CW | m by d | matrix |
| q | 完全な観測保存に必要なランク | 1 | rank(W_c) | 0<=q<=min(d,m-1) | integer scalar |
| Q,P | W_c行空間の直交基底と射影 | 1 | d by q; P=QQ^T | orthonormal Q; rank-zero case uses empty Q | matrices |
| k,P_k | 任意の候補次元と直交射影 | 1 | rank(P_k)=k; d by d | 0<=k<=d | integer, matrix |
| h_*,h_k | 理想補正状態 | 1 | h_0+Pr; h_0+P_k r | oracle quantities | vectors |
| h_hat | 学習済み最終写像の状態 | 1 | d-vector | need not equal ideal state | vector |
| z_T,z_*,u,v | 教師、理想、任意のlogit | 1 | z_T=Wh_T+b,z_*=Wh_*+b | finite m-vectors | vectors |
| c | 全クラス共通logitずれ | 1 | z_*-z_T=c1_m | input-dependent real value | scalar |
| s | softmax | 1 | s(z)_i=exp(z_i)/sum_j exp(z_j) | i,j index classes1..m | vector-valued function |
| e_p,e_f,e | 射影、学習、合計のlogit誤差 | 1 | W_c(h_k-h_T),W_c(h_hat-h_k),W_c(h_hat-h_T) | m-vectors | vectors |
| gamma,epsilon | 教師top1-top2差、誤差最大絶対値 | 1 | teacher margin; max_j abs(e_j) | gamma>0 for unique top1 | scalars |

## Necessary and sufficient condition for exact probabilities for every residual

The condition is W_c P = W_c.

Sufficiency:
1. h_*-h_T=(P-I_d)r.
2. C(z_*-z_T)=CW(P-I_d)r=W_c(P-I_d)r=0.
3. The nullspace of C is span(1_m), since C v=0 implies v=(1_m^T v/m)1_m.
4. Therefore z_*=z_T+c1_m.
5. For each class, exp(z_T,i+c)/sum_j exp(z_T,j+c) cancels the common exp(c) factor and equals s(z_T)_i. All pairwise logit differences and the argmax set are unchanged.

Necessity:
1. If s(u)=s(v), then s(u)_i/s(u)_j=exp(u_i-u_j)=exp(v_i-v_j) for every i,j.
2. Injectivity of the real exponential gives equal pairwise differences; u-v is a common class shift, so C(u-v)=0.
3. Probability preservation for every residual therefore requires W_c(P-I_d)r=0 for every real r.
4. A matrix annihilating every vector is zero, hence W_cP=W_c.

## Minimal exact observable rank

1. Every row of W_c lies in its own row space. Orthogonal projection onto that space leaves every row unchanged, so W_cP=W_c.
2. Conversely any preserving rank-k projector satisfies rank(W_c)=rank(W_cP_k)<=rank(P_k)=k.
3. Thus the minimum projector rank preserving every residual's class probabilities is exactly q=rank(W_c).
4. C annihilates the class-common direction and is identity on its orthogonal complement, so rank(C)=m-1.
5. Consequently q=rank(CW)<=min(d,m-1). With ten classes this gives an upper bound9. Numerical rank9 is tested, not assumed as a universal fact.

This does NOT say that the minimum useful dimension for approximate finite-task accuracy is9. Nor does it say a learned model realizes the oracle.

## Learned error and margin check

Linearity gives e=e_p+e_f. Expanding the inner product gives exactly

    ||e||^2 = ||e_p||^2 + ||e_f||^2 + 2 e_p^T e_f.

The same identity holds after averaging over samples. Its cross term can have either sign. The full centered-head row space makes the ideal e_p zero, but leaves e_f unconstrained.

For a unique teacher top1 class, each competing top1-minus-other logit gap changes by at worst -2epsilon. Therefore gamma>2epsilon suffices to preserve the argmax. A calibration-only check of the IDEAL projection error does not bound the learned e_f on new inputs.

## Shape and unit check

W_cP has shape (m by d)(d by d)=m by d. Projecting r yields a d-vector; applying W_c yields an m-vector. Error squared norms and the cross term have the same units. No speed, byte count, energy or communication result follows from this algebra.

## Exact small example

Take d=2,m=2,W rows(1,0),(-1,0),b=(0,0),h_0=(0,0),r=(2,100),P=diag(1,0). The teacher state(2,100) and corrected state(2,0) both produce logits(2,-2), hence identical probabilities. Their hidden squared error is10000. Hidden-state fidelity and output-task fidelity are different requirements.

## C75E result and boundaries

The source-fixed study used16 fresh model seeds and27 matched final-model conditions. The centered-head margin screen chose9 before fitting in every seed; it met the empirical paired accuracy/NMSE thresholds in16/16. The aggregate head9-versus-P64 accuracy difference was -0.092594pp,95%[-0.486113,+0.324073]; the hidden-NMSE geometric ratio was1.131204,95%[1.111343,1.150885]. Both preregistered non-inferiority gates passed, conditional on the valid P64/direct references.

The 95%-residual-energy screen selected64 in every family/seed: it was conservative rather than useful at reducing dimension. Head8 and residual-SVD8 also passed aggregate candidate gates, so head9 is not proven minimal or superior to all bases. The reverse-order QR16 contrast was -0.856481pp,95%[-1.342592,-0.416667] relative to identity, but this is an exploratory, unadjusted contrast requiring fresh confirmation.

At head9, the ideal centered-logit projection MSE averaged about2.446e-14, while the learned/final-fit logit error MSE averaged0.813673. Exact oracle sufficiency does not remove fitting/generalization loss.

Important:16/16 observed success must not be converted into a claim of100% population reliability. A percentile resample of all-success indicators is degenerate; C76R separately reports an exact binomial interval under an explicit conditional iid-seed interpretation.

The original outputs and source are retained unchanged. The followup C76R locks head-rank selection before fresh outcomes. Complete teacher residuals are still computed at calibration; the current implementation does not demonstrate lower communication cost. No CNN/SmallViT, independent-dataset or physical-hardware claim.

## Primary implementation source

- `scripts/recursive_composition/exploration/c10_boundary_signal_ablation.py`: fixed Linear(64,10) head.
- `scripts/gaussian_shift_interface/c75e_geometry.py`: pre-fit screens, ideal checks and error decomposition.
- PyTorch QR documentation: https://docs.pytorch.org/docs/stable/generated/torch.linalg.qr.html (sign ambiguity and independent-prefix requirements).

## ERROR CHECK

Bias cancellation, common-shift invariance, rank-zero case, minimal-rank necessity, argmax ties, matrix shapes, SI units, float64 oracle versus float32 training, empirical versus population success, and the oracle-versus-learned distinction were checked. This is not independent scientific peer review.
