# Canaria：小規模 digits での合成置換と、その限界

> **整理用の入口（2026-10-29）：** [暫定判断・成果と失敗・今後の条件](../../RESEARCH_CHECKPOINT.md)。本報告の数値・実験結果は変更していません。**固定 test 450 件は過去に評価済みで、未使用の確認集合ではありません。** 新 matched/grid がそれを予測・評価に使わなかったことは、履歴上の既視性を解消しません。[既存プロトコル](../../research/public_checkpoint_20261027/research/digits_matched_regime/PROTOCOL.md)

**研究報告・2026-10-27。** 学習済みネットワークの連続する二つの写像を、一つの写像で近似する方法を検討した。既視 digits の固定条件では、単一写像 C は、共同学習する二写像 J より小さい置換部分の候補予算で品質条件に達する傾向を示した。一方、**全モデル保存量の差は約 1.5% にとどまり、CIFAR-10 では目標の精度維持に失敗した。** 未見データへの一般化や、実験全体の独立再現を示す報告ではない。

本報告は [公開済み証拠スナップショット（commit `661f1c0`）](https://github.com/Unjuno/canaria-neural-simplification/tree/661f1c0648c510ab9303649173dda2bc0997e39e/research/public_checkpoint_20261027) の整理と、同じ保存スカラー結果の追加感度解析である。正・負の結果と再計算可能な範囲をまとめ、新たなデータ取得・学習・推論・速度測定は行っていない。

## 完了している範囲と、未完了の範囲（2026-10-27）

**条件を限定した研究報告は公開済みで、共有できる。一方、実データでの全工程の独立再現・独立した一般化確認まで完了した研究、または main の正式リリースとは位置付けない。**

- **公開ソース・分析：** [保存結果と再計算解析](https://github.com/Unjuno/canaria-neural-simplification/blob/06301e653b69151107af39100e0d4a82445867c5/research-checkpoints/2026-10-27/README.md)に加え、matched・予算グリッド・全モデル保存量・latency のソース、および[新規出力の選定アダプター](https://github.com/Unjuno/canaria-neural-simplification/blob/06301e653b69151107af39100e0d4a82445867c5/research/selected_whole_storage_source_20261027/FRESH_SELECTION.md)と[fresh latency 入力アダプター](https://github.com/Unjuno/canaria-neural-simplification/blob/06301e653b69151107af39100e0d4a82445867c5/research/selected_latency_source_20261027/FRESH_INPUTS.md)まで実装・合成テスト確認済み。ただし、人工入力での確認であり、公開ソースを用いた実データ・全 16 教師・全工程の新規実行は未実施である。
- **原保存物の限定確認：** 公開パッケージに同梱していない原保存物について、選定済み状態 **48/48**、全モデル保存アーカイブ **96/96**、参照ファイル **47/47** の存在・hash 一致を確認した。さらに、選定済み重みと両保存形式のテンソル同一性、選定済み保存 logit **48/48 経路**の形状・型・有限値、および保存ラベル順序・分割 index の整合を確認した。この 47 ファイルは原保存物の内訳であり、新規出力の固定ファイル数ではない。
- **その確認で証明していないこと：** 今回の保存物照合では推論を行っておらず、重みから保存 logit が実際に得られることを改めて検証したわけではない。ラベルと元データ行との意味的対応も独立には再構成していない。原保存物の同一性確認は、新規パイプラインの全工程検証、未見データへの一般化、掲載した科学的結果すべての新しい検証を意味しない。**CIFAR-10 の品質条件未達も変わらない。**

## 1. 比較したもの

- **L：局所学習の二写像。** 各写像を教師の中間出力に合わせて別々の局所損失で学習する。
- **J：共同学習の二写像。** 二写像をつないだ終点の誤差で共同学習する。L だけでなく、この強い比較対象を含める。
- **C：単一写像による合成置換。** 同じ終点を一写像で近似する。J とは構造・パラメータ化が異なる。

digits では 64 次元・4 ブロックの residual MLP 教師の前半 2 ブロックを置換し、stem・後半・分類 head は固定した。既視の学習 1,077 件・validation 270 件と 16 教師を用いる。新 matched-regime の L/J は初期重み・minibatch 列・600 更新を一致させ、固定 2,048 重みで比較した。C も同じ minibatch 列と更新数を使うが、形状が異なるため L/J と同一初期重みにはならない。同じ更新数は同じ FLOPs・学習時間を意味しない。[方法](../../research/public_checkpoint_20261027/research/digits_matched_regime/PROTOCOL.md)

品質条件は **validation の終点 NMSE ≤ 0.08、かつ分類精度が同じ教師から 2 percentage points（pp）以内の低下**の両方。続く予算比較では 512 / 1,024 / 1,536 / 2,048 / 3,072 / 4,096 / 6,144 重みの固定候補をすべて評価した。全 224 セルを残し、うち 32 セルは先の結果の再利用、192 セルが当時の新規 fit である。既視 validation を使った探索的拡張であり、独立した確認用データではない。[予算比較の方法・全結果](../../research/public_checkpoint_20261027/research/digits_joint_budget_grid/results/REPORT.md)

## 2. 主な結果

以下の主結果表の比率は 16 教師の対応付き幾何平均。掲載した 95% 区間は教師単位の対応付き bootstrap 100,000 回で、固定済みデータと教師 recipe に条件付けた区間である。追加のしきい値感度解析には新たな信頼区間を付けていない。

| 問い | 結果 | 読み取れる範囲 |
|---|---|---|
| 強い J を比較対象にする必要があるか | 固定 2,048 重みで、平均 log2(NMSE J/L) は −1.168441、95% 区間 [−1.247279, −1.092253]。J−L の平均精度差は +1.7593 pp。品質合格は L 0/16、J 12/16、C 15/16 | L/J の学習方式の比較。teacher forcing と勾配経路も変わるため、目的関数だけの因果効果ではない |
| C は J より少ない置換予算で合格するか | 候補内の初回合格予算比 C/J = **0.788324**、95% 区間 **[0.694450, 0.904368]**。C が小さい 11、同じ 4、大きい 1 | 置換部分の初回合格候補予算比の対応付き幾何平均として約 21.2% 少ない。各教師で一律に減ったわけではない。理論最小容量ではない。3 系列の非単調な合否も残した |
| 全モデルも同じ割合で小さくなるか | C/J 全パラメータ比 **0.987441**、NPZ 実 bytes 比 **0.984674**（非圧縮）/ **0.984369**（DEFLATE） | 全体の差はパラメータ約 **1.26%**、保存量約 **1.53–1.56%**。RAM や配備パッケージの削減率ではない |
| 選定済み全モデルの forward は速いか | C/J CPU 時間比は batch 1 で **0.836712**、batch 128 で **0.943339**。両 batch とも C が短い 16/16 | この測定条件では約 16.33% / 5.67% 短い。他機器での速度保証ではない |

出典：[matched-regime](../../research/public_checkpoint_20261027/research/digits_matched_regime/results/REPORT.md)、[予算グリッド](../../research/public_checkpoint_20261027/research/digits_joint_budget_grid/results/REPORT.md)、[全モデル保存量](../../research/public_checkpoint_20261027/research/selected_whole_storage/REPORT_JA.md)、[CPU 時間](../../research/public_checkpoint_20261027/research/selected_latency/REPORT_JA.md)。

CPU 測定は固定 synthetic 入力、共有ホスト、eager CPU・1 thread、batch 1/128 に限定される。16 seeds × 2 batches × 3 arms × 30 rounds = **2,880 timing blocks**、各 block は 20 forwards。各 seed の全 30 blocks の中央値から比率を求めた。最大で同セル中央値の 25.235 倍の block も削除していない。ロード等の時間は含まず、構造と選択幅の両方が異なるため「深さを減らしたことだけの効果」とはしない。[測定条件とばらつき](../../research/public_checkpoint_20261027/research/selected_latency/REPORT_JA.md)

### 保存済み結果のしきい値感度：方向と選定の不安定さ

同じ既視 validation の全 224 セルを用い、NMSE 上限 0.06 / 0.07 / 0.08 / 0.09 / 0.10 と、教師より多く誤ってよい件数 3 / 4 / 5 / 6 / 7 件（270 件中）を組み合わせた **25 条件**を追加点検した。元の 2 pp 条件の合否は、この標本では「追加誤り 5 件まで」と一致する（5 件自体は約 1.852 pp であり、2 pp と数値的に等しいわけではない）。これは結果観察後の **post-hoc 記述解析**であり、元の判定条件・選定モデル・結論は変更しない。

**全 16 組で比を計算できた 20 条件**の置換部分の幾何平均候補予算比 C/J は **0.676555–0.891150** で、いずれも 1 未満だった。しかし、追加誤り 3 件までの **残り 5 条件は打ち切り**となった。教師 ID 1 の C は候補上限 6,144 まで合格せず、J は 6,144 で合格したため、全体比を計算せず、完了例だけの集計にも置き換えていない。「25 条件すべてで優位」とは言えない。

元の条件では **0.788324、11 小・4 同・1 大**を再計算できた。一方、元の選定で **C、J とも各 3/16** は正解件数の余裕がゼロで、教師の正解数などを固定して正解が 1 件減れば精度条件に不合格となる。NMSE 上限 0.08 のまま許容追加誤りを 5→4 件にすると計 6 選定、5→6 件にすると計 7 選定が変わる。これは不確実性の推定ではなく、判定境界からの距離の記述である。**集約した方向がこの範囲で維持されても、個別の選定幅と厳しい条件での全組合格は安定とは限らない。** 新データ確認、20 回の独立再現、任意のしきい値での優位、代替しきい値の新しい信頼区間を示すものではない。[25 条件すべての結果と方法](../../research/threshold_robustness_20261027/REPORT.md)

## 3. CIFAR-10：成功していない結果

元の Comp は 3 教師に対して validation 精度を **3.70 / 3.54 / 3.86 pp** 下げ、既に実施済みの official test 10,000 件でも **4.26 / 4.22 / 4.38 pp** 下げた。3 教師 × 3 比較手法の全 9 arm が「教師からの低下 ≤ 2 pp」の条件に不合格だった。比較対象より良い場合があっても、要求品質の達成にはならない。[結果一覧](../../research/public_checkpoint_20261027/docs/claims_checkpoint_20261027/README.ja.md)、[official test の保存結果](../../research/public_checkpoint_20261027/research/cifar10_heldout/execution/results.json)

幅・学習予算・KL、空間配置/dilation、decision-preservation の検査した改善系列も所定の条件を満たさなかった。この介入系列は負の結果として報告するが、あらゆる合成置換が不可能だという証明ではない。既視 official test を再調整や新しい確認に転用しない。[負の結果と解釈の限界](../../research/public_checkpoint_20261027/docs/research_checkpoint_20261026/SCIENTIFIC_APPENDIX.md)

## 4. この報告が示していないこと

1. **未見データ・別 writer への一般化。** 別筆者データによる確認は、データ取得が成立せず未実施。結果がないことを合格と扱わない。同じ文字データ系統の別 writer を検証できたとしても、任意の独立データセットへの一般化とは別である。
2. **実験全体の独立再現。** 保存済み数表・ログ・hash の照合、別担当による証拠監査、同じ cohort の技術的再実行、新しい独立標本での再現は別物である。今回の公開資料は最後のものを提供しない。
3. **全モデル半減・普遍的な圧縮則。** 旧 R87R2 の C/L = 0.531610 と Phase4 の 0.476029 はそれぞれの選択部分の予算比であり、全モデルや他構造の半減を意味しない。旧 C/L = 0.5316 と新 C/J = 0.7883 の差は cohort・recipe も異なり、比較対象強化だけの効果に分解できない。旧 Issue87 の数値・来歴問題も未解決のまま残す。
4. **任意の配備環境での費用削減。** 固定 NPZ の保存量と共有 CPU の forward 時間のみを区別して報告する。旧コスト pilot の「bytes と時間をともに 5% 以上削減」が不成立だった結論を、今回の別モデル・別比較で上書きしない。

これらは主張を広げるために残る確認事項であり、条件を限定した現状報告そのものを成功談に書き換えたり、無期限の追加実験待ちにしたりする理由にはしない。[主張と証拠の対応](../../research/public_checkpoint_20261027/docs/claims_checkpoint_20261027/README.ja.md)

## 5. 読者が再計算できる範囲

固定 commit `661f1c0` の旧チェックポイント更新は **70 個のテキストファイルによる部分抜粋**であり（今回追加する感度解析を含む総ファイル数ではない）、元の完全な学習・推論・計測 runner、全モデル重み、画像・データを含む再現パッケージではない。スカラー再計算コードは別実装で、原実験を再実行するものではない。この旧スナップショットの公開時の記録では、保存数表の **29 集計項目が完全一致し、12 件の合成テストに合格**している。これは新しい科学的試行数ではない。[公開資料の範囲](../../research/public_checkpoint_20261027/README.ja.md)、[再計算コードと検証記録](../../research/public_saved_analysis_20261027/README.md)

Python 3.11 以降・NumPy 2.3.5 の環境で、この追加解析を含む更新済み `research/checkpoint-20261026` ブランチまたは同じ内容の公開パッケージを取得し、そのルートから実行する。上記 `661f1c0` は元資料を固定するための旧スナップショットであり、新しい `threshold_robustness_20261027` ディレクトリは含まない。以下の再計算自体にネットワーク、torch、モデル、データ、資格情報は不要である。

```sh
cd research/public_saved_analysis_20261027
python -m unittest -v
python check_checkpoint.py ../public_checkpoint_20261027
```

個別の再計算例：

```sh
python analyze_saved.py matched ../public_checkpoint_20261027/research/digits_matched_regime/results/SUMMARY.json
python analyze_saved.py grid ../public_checkpoint_20261027/research/digits_joint_budget_grid/results/evidence/SUMMARY.json
python analyze_saved.py storage ../public_checkpoint_20261027/research/selected_whole_storage/results/ROWS.csv
python analyze_saved.py latency ../public_checkpoint_20261027/research/selected_latency/results/RAW.jsonl
```

追加のしきい値感度解析は [別の再計算手順](../../research/threshold_robustness_20261027/README.md) に従う。こちらは Python 標準ライブラリだけを使い、同梱のスカラー入力から全 25 条件と選定余裕を再計算できる。上のディレクトリからは次のように実行する。出力先は存在しないディレクトリを指定する（既存結果は上書きしない）。

```sh
cd ../threshold_robustness_20261027
python -m unittest -v
python analyze_thresholds.py SCALAR_INPUT.json reproduction
```

入力欠落や不正値を除外して成功例だけを集計せず、エラーとして扱う。予算グリッドで一部の教師・手法に合格候補がなければ、到達範囲を報告し、完了例だけの全体比は作らない。再計算対象の原資料・変換内容・hash は [PROVENANCE.json](../../research/public_checkpoint_20261027/PROVENANCE.json)、[PATH_HASH_MANIFEST.json](../../research/public_checkpoint_20261027/PATH_HASH_MANIFEST.json)、[コードの変換説明](../../research/public_saved_analysis_20261027/TRANSFORMATION.md) を参照。hash の一致は来歴の全保証や独立した実験成功を意味しない。

追加公開：[matched L/J/C のソース一式](../../research/digits_matched_source_20261027/README.md)。別途の独立レビューで合成テスト 27/27 に合格したが、元のモデル・配列・データは含まず、実データによる科学的再実行・独立再現は未実施である。パッケージ内の「実行未検証」等は先行する梱包時点の記録であり、その後の合成テスト合格は full-cohort 実行や数値結果の再現を保証しない。

追加公開：[7 予算 J/C グリッドの source-only パッケージ](../../research/digits_joint_budget_source_20261027/README.md)。独立レビューで 40 種類の合成テスト（親契約 19・runner 21）に合格した。レビュー側の検査処理を補正した後にソース検査 1 件を再確認したため計 41 回実行し、生成テンソルでの optimizer 更新は計 3 回に限られる。モデル・配列・実データ・元の非公開アーカイブは含まず、完全な新規 matched 出力を親として用いる新しい技術的再実行向けのソースであり、原アーカイブの復元ではない。実データによる full-cohort 再実行・数値結果の再現は未実施。パッケージ内の「実行検証待ち」等は梱包時点の記録であり、今回の合成テスト合格とは区別する。

追加公開：[選定済み全モデル保存量の source-only パッケージ](../../research/selected_whole_storage_source_20261027/README.md)。独立レビューで合成テスト計 21 件（同梱 15・追加 6）に合格した。元の重み・配列・データ・選定マニフェストは含まず、全モデル保存量の再測定・原実験の独立再現は未実施である。パッケージ内の「実行未検証」等は梱包時点の記録であり、今回の合成テスト合格は実モデルでの再測定や保存 bytes の再現を保証しない。

追加公開：[選定済み全モデル latency の source-only パッケージ](../../research/selected_latency_source_20261027/README.md)。独立レビューで合成テスト 15 件に合格した。元のモデル重み・配列・データ・非公開マニフェストは含まず、実モデルの実行・新たな時間測定・原実験の独立再現は未実施である。パッケージ内の「実行未検証」等は梱包時点の記録であり、今回の合成テスト合格は実モデルの正確性や時間測定の妥当性を保証しない。

## 結論

**固定された小規模 digits 条件では、強い共同学習 J に対しても、単一写像 C の置換部分の初回合格候補予算と条件付き CPU 時間に利点が見られた。ただし、全モデル保存量の差は小さく、CIFAR-10 の品質条件は未達だった。** ここまでを、再計算可能な数表と未確認事項を併記した研究結果として公開する。

