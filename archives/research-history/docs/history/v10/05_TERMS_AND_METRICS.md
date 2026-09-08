# 用語・指標

## カナディア / Canary
会話上の呼称。現在は「境界で観測されるstress/sensitivity signal」として扱う。根本現象そのものとは仮定しない。

## Boundary expansion
内部spanの境界を外側へ移動し、複数実装blockを一つの機能spanとして扱うこと。

## Compile
learned neural spanをexplicit operator/functionへ置換すること。

## Freeze
compiled operatorを固定し、周囲のみ再学習すること。

## Recontract / recontracting
固定された新しいboundary contractのもと、周囲のNNが再適応すること。

## Simplification complexity C(S)
指定utility基準を満たすspan Sの最小明示記述量。parameterだけでなく、保存が必要なbuffer/係数も含める。

## Composition gain
\[
G(A,B)=1-\frac{C(A\cup B)}{C(A)+C(B)}
\]
G>0で劣加法的簡約。

## Intrinsic simplification
周囲を再学習せず、元span写像を忠実に再現しただけでCが減る現象。

## Adaptive simplification
小さいoperatorを固定後、周囲の再学習でutilityが回復し、結果として小さいCで成立する現象。

## Mechanism inheritance
subspace similarity / Jacobian similarity / affine residual等で、前機構の骨格が次状態にどれだけ残るかを測る。

## Archetype
実装名ではなくbehavior/fingerprintで同一視した粗い機構型。

## Mechanism manifold / chart
一つのarchetype内部で連続的に変化する低次元parameter領域。

## Participation ratio
PCA固有値から計算するeffective dimensionの一指標。

## Gauge / alignment
内部channel basisの回転・置換・scalingなど、機能を変えず表現座標だけを変える自由度、およびその整合。

## 主要評価軸
- clean accuracy
- augmented/shifted accuracy
- utility vs matched control
- core parameter/storage ratio
- fit relative MSE
- Canary / shell gradient ratio
- prediction agreement
- Jacobian cosine
- operator subspace similarity
- affine-remap residual
- inheritance score
- PCA 90/95/99% dimension
- archetype ID
- repair epochs tau
- simplification gain G
