# Phase A Confirmatory Protocol — Canary-blind Simplification Map

## 0. 目的
最重要仮説を、探索時にCanaryを参照しない形で検証する。

対象は「Canaryがsimplificationの必要条件か、それとも探索効率を高める部分観測sensorか」である。ここで得られるのは **小型CNN内での反証可能な判定** であり、全NNへの普遍則の証明ではない。

## 1. 変数表

| 記号 | 意味 | SI単位 | 定義 | 定義域 / 前提 | 型 |
|---|---|---|---|---|---|
| `S` | 連続span | 無次元 | block index区間 `[i,j]` | 6–8 block CNN内 | 離散区間 |
| `A,B` | `S` を分割した隣接subspan | 無次元 | `S=A∪B`, `A∩B=∅` | contiguous binary split | 離散区間 |
| `τ` | repair budget | epoch | `{0,1,2,4,8}` | matched controlも同epoch追加学習 | 離散スカラー |
| `C(S;τ)` | utility基準を満たす最小保存量 | parameter-equivalent count | learned parameter + learned buffer + learned scalar | primary utility floor=0.95 | 非負スカラー |
| `G(A,B;τ)` | composition simplification gain | 無次元 | `1 - C(A∪B;τ)/(C(A;τ)+C(B;τ))` | 分母>0 | 実スカラー |
| `U` | matched-control utility ratio | 無次元 | compiled augmented accuracy / matched-control augmented accuracy | control accuracy>0 | 非負スカラー |
| `Y` | strong simplification indicator | 無次元 | `1[G≥0.25 and U≥0.95]` | primary confirmatory label | Bernoulli |
| `K` | Canary scalar | 無次元 | frozen sensor definitionのboundary stress ratio | simplification測定後に計測 | 非負スカラー |
| `K_pct` | width-conditioned Canary percentile | 無次元 | `(seed,state,width)` 内rank percentile | 0–1 | 実スカラー |
| `L` | low-Canary indicator | 無次元 | `1[K_pct≤0.25]` | width confoundingを一次除去 | Bernoulli |
| `H` | high-Canary indicator | 無次元 | `1[K_pct≥0.75]` | 同上 | Bernoulli |
| `q_L` | low-Canary strong simplification率 | 無次元 | `P(Y=1 | L=1)` | seed-clustered inference | 確率 |
| `ΔAUC` | Canary追加予測情報 | 無次元 | `AUC(width+K)-AUC(width only)` | leave-one-seed-out | 実スカラー |

### 次元チェック
`C` は同じ保存量単位なので `C(A∪B)/(C(A)+C(B))` は無次元。したがって `G`, `U`, `K_pct`, `q_L`, `ΔAUC` はすべて無次元で整合する。

## 2. H — 反証可能仮説

### H-A: Canary-local 仮説の反証
Canaryが低い領域ではstrong simplificationが実質的に存在しない。

- Canary-local予測: `q_L ≤ 0.05`
- Canary-observer予測: `q_L` は非自明に正であり、設計上の実用閾値として `q_L ≥ 0.15` を候補とする。

`0.05/0.15` は自然定数ではなく、事前に固定する **decision margin** である。

### H-B: Canaryは追加情報を持つ
widthだけに対してCanaryを加えると、seed holdout予測が改善する。

- `ΔAUC > 0`
- ただし `q_L > 0` が同時に成立するなら、「必要条件」ではなく「部分sensor」と解釈する。

### H-C: 小型CNN内のcomposition subadditivity
confirmatory seedsで、Canaryを使わず生成したcomposition eventsの過半数で `G>0` が成立する。

これは **within-architecture statistical law** の検定であり、cross-dataset universal lawとは呼ばない。

## 3. T — 最小検証

### Model / data
- 8-block CNNを第一選択。計算制約が強い場合のみ6-block。
- datasetは既存digits系を使うが、split hashを保存。
- candidate grammarとcomplexity definitionをrun開始前にfreeze。

### Seeds
- confirmatory minimum: `n_min = 8 model seeds`。
- fixed seed list: `1000..1007`。
- 判定不能なら `1008..1011`、さらに必要なら `1012..1015` を追加。
- `n_max = 16`。追加seedの途中でdecision marginやgrammarを変更しない。

### Blindness barrier
**Stage 1: simplification only**
1. 全contiguous spanとbinary splitを列挙。
2. `τ={0,1,2,4,8}` でcandidate評価。
3. `C`, `G`, `U`, `Y` を保存。
4. Stage-1結果にCanary列を含めない。
5. Stage-1 table SHA256を固定。

**Stage 2: sensor only**
1. Stage-1 table hash固定後にCanaryを計測。
2. `event_id` だけでjoin可能な別tableへ保存。
3. Canary測定コードはcandidate選択・repair・early stopに影響しない。

**Stage 3: locked join**
- Stage 1とStage 2を `event_id` でjoinして初めて `q_L`, `ΔAUC` を計算する。

### Baselines
1. matched continued-training no-compile control
2. random span search
3. Canary-guided search（**Stage 3後の探索効率比較だけ**）
4. target-specific best explicit operator
5. shell-capacity restricted variant（Phase Bへ接続）

### Statistics
- eventを独立標本として扱わない。
- primary CIは **seed-cluster bootstrap**。
- predictionは leave-one-seed-out。
- width-conditioned percentileでCanaryのscale confoundingを抑える。
- 90/95/98/100% utility floor sensitivityはsecondaryとして保存する。

## 4. D — 合否条件

### D-A: Canary-local vs observer
- **PASS observer / FAIL local**: `lower95CI(q_L) > 0.10`
- **PASS local / FAIL observer**: `upper95CI(q_L) < 0.05`
- **UNCERTAIN**: 上記以外

0.10は0.05と0.15の事前margin中央で、判定の安定性を確保するための境界。

### D-B: Canaryの追加sensor価値
- **PASS**: `lower95CI(ΔAUC) > 0`
- **FAIL**: `upper95CI(ΔAUC) ≤ 0`
- **UNCERTAIN**: CIが0を跨ぐ

### D-C: within-architecture subadditivity
`q_G = P(G>0)` とする。

- **PASS**: `lower95CI(q_G) > 0.50`
- **FAIL**: `upper95CI(q_G) ≤ 0.50`
- **UNCERTAIN**: 0.50を跨ぐ

### Sequential stopping
`n=8,12,16` seedsの時点だけ判定する。

- primary D-AとD-Cが両方decisiveなら停止可能。
- どちらかがUNCERTAINなら次の4 seedsを追加。
- `n=16` でもUNCERTAINなら「未確定」と報告し、閾値を後付けで変更しない。

## 5. C — 破れ方 / 対立仮説

1. **Width confounding**: Canaryがwidthの代理変数にすぎない。
   - 対策: width-conditioned percentile + width-only baseline。
2. **Shell overcapacity**: simplificationではなくshellが失われた計算を吸収している。
   - 対策: shell capacity restriction。
3. **Repair regularization**: compileが効くのでなく追加学習が効いている。
   - 対策: matched no-compile continuation。
4. **Finite grammar bias**: `C` がcandidate grammarの制約を反映しているだけ。
   - 対策: grammar version固定 + later grammar expansion sensitivity。
5. **Nested-event pseudo-replication**: overlap spanを独立としてCIが過度に狭い。
   - 対策: seed-cluster bootstrap、parent-span cluster sensitivity。
6. **Sensor leakage**: Canary値がspan/candidate選択に暗黙利用される。
   - 対策: Stage 1/2 process分離、immutable hash、別output。

## 6. U — 不確かさ

主要誤差源:
- seed間変動
- dataset split変動
- repair stochasticity
- candidate optimization variance
- overlapping spansの依存
- finite grammar truncation
- Canary sensor calibration drift

合成不確かさは単純な独立誤差伝播より、cluster bootstrap分布を主とする。報告は95% percentile CIをprimaryとし、必要ならBCa bootstrapをsecondaryとする。coverage factor `k=2` を機械的に当てるより、依存構造を保ったresamplingを優先する。

## 7. 成果物

必須:
- `stage1_blind_simplification.csv`
- `stage1_blind_simplification.sha256`
- `stage2_canary_sensor.csv`
- `stage3_joined_analysis.csv`
- `run_metadata.json`
- `candidate_grammar.json`
- `decision_report.md`
- 全script SHA256

## 8. 重要な解釈制限
Phase Aでobserver仮説がPASSしても、「ニューラルネット全体で普遍的に成立」が証明されたわけではない。次段階でFashion-MNIST、CIFAR-10、deeper CNN/ResNetへ外的妥当性を拡張する必要がある。
