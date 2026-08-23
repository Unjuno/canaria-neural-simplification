# 研究引き継ぎ — カナディア / Canary / Neural-to-Explicit-Function Compilation

## 1. 研究目的の変遷
当初の目的は「Canary信号を使ってNNのどこが圧縮可能かを見つける」ことだった。研究が進むにつれ、次のより一般的な問題へ発展した。

> ニューラルネット内部の分散した計算を、自然な機能境界でまとめ、少数種類・低次元・再利用可能な明示的機構へ反復的にコンパイルできるか。

現在の研究対象はCanaryそのものではなく、その上流にある可能性のある **neural computational composition and simplification law** である。

## 2. 現在の統一モデル
spanを
\[
S=B_{out}\circ C\circ B_{in}
\]
とし、内部を明示機構 \(\tilde C\) へ置換する。

現在の実験を最もよくまとめるモデルは
\[
C_{span}\approx A_{out}\circ \Phi_k(\theta)\circ A_{in}+R
\]
である。

- \(k\): mechanism archetype ID
- \(\Phi_k\): 共有可能な機構原型
- \(\theta\): 2〜数個程度の低次元座標候補
- \(A_{in},A_{out}\): representation / gauge alignment
- \(R\): まだ吸収されていないresidual mechanism

## 3. Canaryの現在の位置付け
初期にはCanaryを「圧縮可能性」または「安全性」とみなしたが、これは強すぎた。

現在の最良の解釈：
\[
\boxed{\text{Canary} \approx \text{boundary-contract stress / sensitivity}}
\]

代表指標：
\[
A_{canary}=\frac{\|\nabla_{B_{out}}\mathcal L_{after}\|}{\|\nabla_{B_{out}}\mathcal L_{before}\|+\epsilon}
\]

重要な分離：
- Compatibility: 境界に適合できるか
- Adoption: 学習がその機構を使うか
- Recontracting: 周囲がどう再適応するか
- Utility: 最終性能に役立つか

確認済み：
\[
\text{Compatibility/Adoption} \neq \text{Utility}
\]
および
\[
\text{low Canary} \not\Rightarrow \text{high utility}
\]

Canary suppressionでも圧縮性は自動的に上がらず、Canaryは原因より観測系と考える方が整合的。

## 4. Boundary expansion / span merging
実装block境界をそのまま自然な機能境界と仮定せず、境界を外側へ移動し複数blockを一つのspanとして扱う。

主要結果：
- 3-block full span → single linear 3x3 + FT で baseline同等以上、core ratio約0.167
- 4-block full span → single linear 3x3 + repair で core ratio約0.125
- 個別block置換が失敗してもfull-span単一関数が成立することがある

したがって：
\[
\boxed{\text{implementation block boundaries} \neq \text{natural functional boundaries}}
\]

## 5. 関数族探索
full-span fitで観測された例：
- linear7x7 rel MSE ~0.017
- linear5x5 ~0.0246
- linear3x3 ~0.0663
- lowrank linear3x3 r4 ~0.0880
- lowrank r2 ~0.1299
- nonlinear bottleneck r4 ~0.1434
- linear1x1 ~0.2372

機構としては「low-rank channel-mixed spatial linear operator」が有力だった。ただし反復recontracting後にはdepthwise+pointwiseが有効になるケースもあり、機構族自体が学習状態に依存して遷移する。

## 6. 反復コンパイル
実装済みループ：
1. span/probe
2. explicit operator fit
3. freeze
4. remaining network retrain
5. re-probe
6. recompile
7. automatic stop

手動段階例：
- 4 learned blocks
- block0–1 → K1
- K1 + block2 → K2
- K2 + block3 → K3
- core ratio 1.0 → .625 → .375 → .125
- matched augmentation性能維持

自動化：
- fixed Canary threshold
- utility trial
- Pareto knee
- 4-objective Pareto (compression, utility, fit, Canary)
- automatic function-family selection
- finite expression grammar synthesis

重要：固定閾値を外しても自律選択・停止は可能。ただしPareto正規化等のmeta-ruleは人間設計。

## 7. 機構継承と新機構形成
same-taskでcompile→freeze→retrainすると、完全な別関数へ飛ぶより、主要operator basisを保ちながら補正が育つ。

代表：
\[
F_{t+1}\approx A_t(F_t)+R_t
\]

4 epochではmechanism inheritance約94–98%、12 epochでも約90%以上のケースが多い。

一方、新タスク（orientation / position classification）へshiftすると、utilityを獲得しながらinheritanceが0.71–0.74程度まで低下し、Jacobian similarity低下・affine residual増加が観測された。

解釈：
- same-task: inheritance + small residual
- new-task: partial inheritance + larger task-specific residual

## 8. 関数種類 atlas
多数のseed/span/state/candidateから置換可能関数を集め、channel gaugeに依存しないfingerprintでクラスタリング。

粗い解像度では約29 clusters、推定飽和約30.6。細かい解像度ではまだ増加。

現時点のモデル：
\[
\mathcal F\approx \bigcup_{k=1}^{K}\mathcal M_k
\]
- \(K\): 有限に近いarchetype数
- \(\mathcal M_k\): 各archetype内の連続parameter manifold

つまり「有限個の完全離散関数」ではなく「少数archetype + low-dimensional variants」。

## 9. Boundary sizeとmanifold dimension
右境界固定でspan width 1→4へ拡張すると、学習trajectory上のeffective operator manifoldが低次元になる傾向。

4-seed集計（病的fitケース除外）：
| width | participation ratio | 95% dimension | rank2 weight error |
|---:|---:|---:|---:|
|1|2.63|4.33|24.8%|
|2|2.37|3.75|13.7%|
|3|2.21|3.75|13.3%|
|4|1.98|2.50|8.26%|

span width vs participation ratio Spearman ~ -0.853。

仮説：
\[
\boxed{\text{larger functional spans may be more compressible in mechanism-coordinate space}}
\]

## 10. Mechanism dictionary / coordinate compression
full-span Conv5は1608 parametersだが、同一seedの学習trajectoryでは95%変動を2〜3座標で説明できた。

\[
K_t\approx K_0+\alpha_{t1}D_1+\alpha_{t2}D_2
\]

rank2 reconstructionでexact Conv5とのaugmentation accuracy差は平均ほぼ0、絶対差約0.6pt程度。

これは「各出現箇所で全weightを保存せず、mechanism ID + few coordinatesで持てる」可能性を示す。

ただしcross-seed global dictionaryは未成功。1x1/3x3 gauge alignmentだけではrepresentation mismatchを十分解消できず、target-specific operatorに届かない。global canonicalization / chart alignmentが主要未解決点。

## 11. Dictionary reuse
shared prototype + tiny 1x1 adaptersで別seedへ転用すると、ケースによってtarget-specific low-rank operatorに近い性能まで回復。random prototypeより良い。

ただしadapterが単なるgauge alignmentでなくresidual correctionも担っている可能性があり、pure universal dictionaryの証明には未達。

## 12. 一般 composition / simplification law
Canaryから独立に、明示表現の最小記述量を \(C(S)\) と定義。
隣接A,Bについて
\[
G(A,B)=1-\frac{C(A\cup B)}{C(A)+C(B)}
\]
を測定。

4 seed, 80 composition events:
- 全体簡約: 72.5%
- width2: 50.0%
- width3: 68.8%
- width4: 100%
- 25%以上簡約: 全体68.8%

簡約したunique parent spanの53.1%は内部にも別の簡約subspanを持った。

したがって再帰的なsubadditivity候補：
\[
\boxed{C(A\cup B;\tau)<C(A;\tau)+C(B;\tau)}
\]

## 13. Intrinsic simplification と Adaptive simplification
重要な分岐：
- intrinsic: 元span写像を忠実にfitした時点で簡単
- adaptive: replacementをfreezeし周囲をretrainすると簡単な解へ移る

full-spanでは即時fitで簡単とは限らないが、repair後には非常に小さなoperatorでutilityが回復。

8 full-span条件の例：
- direct: 0/8が95% utility基準通過
- 4-epoch recontract: 8/8通過
- 保存量平均: 元coreの約1.58%

したがって大span collapseの主成分は単なる代数的function compositionだけでなく **constraint-induced computation redistribution** の可能性が高い。

## 14. Simplification formation time
同じsmall operatorを固定しrepair時間 \(\tau\) を変えた結果：
| tau | 95%到達率 | mean utility |
|---:|---:|---:|
|0|0%|0.602|
|1|0%|0.832|
|2|62.5%|0.949|
|3|62.5%|1.003|
|4|100%|1.045|

静的な簡約ではなく、
\[
P(\text{simplification}\mid w,\tau,context)
\]
として扱う必要がある。

## 15. Canaryは一般則の部分観測か
4-seed composition datasetで、simplification予測のleave-one-seed-out AUC：
- width only ~0.710
- width + scalar Canary ~0.750
- width + Canary response vector ~0.736
- width + Canary + fit response ~0.799

ただしwidth固定ではCanaryの予測力は弱い場合が多い。

解釈：
\[
\boxed{\text{Canary}\neq\text{simplification law}}
\]
より上流にsimplification/composition processがあり、Canaryはboundary stressを通じた部分観測である可能性。

## 16. Knowledge distillationとの接続仮説
蒸留が成功する理由の一部として、teacherのparameter complexityよりeffective mechanism complexityがはるかに低い可能性。

\[
\text{large teacher params}\rightarrow\text{small effective mechanism set}\rightarrow\text{small student}
\]

未検証。今後mechanism-aware distillationを直接比較する価値が高い。

## 17. 研究の現在位置
研究の中心は次の一般仮説へ移行している：

> ニューラルネット全体には、隣接・重複する計算を合成すると適応後の最小実効記述量が劣加法的に減少する領域が広く存在し、Canaryはそのうち境界stressとして観測しやすい場所を検出している。

次の決定的実験は **Canaryを一切使わないblind/exhaustive simplification map** と Canary map の独立比較である。
