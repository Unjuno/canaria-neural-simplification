# Canaria

**ニューラルネットワークの計算を、タスクに必要な振る舞いを保ちながら、より小さな表現・写像へまとめ直す研究。**

[English](README.md) · [研究全体の索引](docs/RESEARCH_INDEX.md) · [Digits / CIFAR 詳細](RESEARCH_CHECKPOINT.md) · [再現状況](STATUS.md) · [リポジトリ構成](REPOSITORY_LAYOUT.md)

ブロックを個別に近似するだけでなく、合成された入出力関係を直接近似すると何が変わるか。学習中の統合、低精度・構造化表現、再帰合成、画像・系列モデル、実行方式まで調べ、**削減できた条件と、品質を保てなかった条件の両方**を保存しています。

## 何を研究してきたか

1. **学習済み計算の置換**：複数ブロックをまとめる写像と、個別・共同学習する写像を比較。置換部分だけでなく、モデル全体のパラメータ・保存 bytes・実行時間を分けて評価します。
2. **学習中の統合**：途中で計算をまとめ直した後の学習・修復を調べます。置換直後と追加学習後の成果を混同しません。[学習中の統合](docs/TRAINING_TIME_CONSOLIDATION.md)
3. **品質・一般化の境界**：特徴誤差が減っても、分類・生成の振る舞いが保たれるとは限りません。目的関数、補正部分空間、再帰構成、データ・モデル系列を変えた実験で、その境界を調べます。

## 研究全体を読む

以下は保存された一次報告・数表に基づく研究の全体像であり、この整理で新しく実測・再現した結果ではありません。最新の Digits / CIFAR 系列だけが本リポジトリの成果ではありません。次の系列は異なるモデル・比較・品質条件を持ち、合算しません。

| 別系列で記録された成果 | 何を削減・改善したか | 条件・証拠の範囲 |
|---|---|---|
| 歴史的 whole-model / codec | v18 の FP32 code **26.138% 減**、q8 zlib code **28.786% 減**。v19 は固定 payload **9,926 bytes**、round-trip logit 差 0 | matched repair 対照・固定 architecture / codec。実行プログラム全体や RAM のサイズではない。[v18](archives/research-history/results/v18/raw/phaseX_summary.json) / [v19](archives/research-history/results/v19/raw_AD/confirmatory_codec_summary.json) |
| SmallViT・synthetic 系列 | v20 SmallViT **60.182%**、v21 synthetic sequence **58.053%**、v22 synthetic causal LM **58.154%** のパラメータ削減 | v20 / v22 の報告上の品質合格は追加修復後。v21 は activation fitting 後・task repair なし。v20 置換直後、v22 置換直後の generation は不合格。[v20](archives/research-history/results/v20/g3_confirmatory_summary.json) / [v21](archives/research-history/results/v21/g5_confirmatory_summary.json) / [v22](archives/research-history/results/v22/g6_confirmatory_summary.json) |
| California Housing | composed/component-wise の選定置換予算比 **0.476029**、test R² 差 **+0.006858** | prospective 8 model-seed、1 固定分割・同じ residual-MLP grammar。[報告](https://github.com/Unjuno/canaria-neural-simplification/blob/f7d23c9dad2788e64b94f2a268fe3e0b1084d77c/results/phase4/california_regression/stage_b_confirmatory/RESULT.md) |
| 学習中の段階的統合 | G7 は同じ最終モデルの early / late 統合より低い PPL、G19 は同じ 192 updates で段階経路 PPL 19.3580 対直接 20.0475 | 小規模 character-LM の記録。追加学習なしの G17 は、直接方式との差が事前指定 ±0.10 PPL 内で同等でした。[学習・対照](docs/TRAINING_TIME_CONSOLIDATION.md)、[後続結果](docs/LATE_STAGE_FINDINGS.md) |
| 方向を含むタスク損傷予測 | G23–G25 は方向・logit 誤差を使う予測と移転、G26 は将来 horizon の補正による改善を記録 | 公開 summary と来歴 hash の照合であり、原観測データ・予測器実装の独立再検証ではない。予測改善は最適 controller の完成を意味しない。[記録](docs/LATE_STAGE_FINDINGS.md)・[原資料の残存範囲](results/training_time/ARTIFACT_INVENTORY.md) |

これらの正の結果と同時に、real-text の小規模 LM では検査した rollout preservation が不合格でした。teacher-forced perplexity の保持を自由生成の保持と読み替えません。精度・修復系列には実装バグで無効化された結果もあり、[訂正履歴](results/phase2/precision_composition/INVALIDATED_HISTORY.md)を残しています。

## 主要な結果：何と比べて、何が小さくなったか

現在の Digits 比較では、4 ブロックの residual MLP の前半 2 ブロックを置換し、残りを保持します。**L** は局所的に学習する 2 写像、**J** は共同学習する 2 写像、**C** は単一写像です。16 教師は同じデータ分割を使い、独立した 16 データセットではありません。品質条件は validation の NMSE ≤ 0.08、教師からの精度低下 ≤ 2 pp です。

| 比較・対象 | 保存済み結果 | 適用範囲 |
|---|---|---|
| **元教師 → C / J：モデル全体** | C はパラメータ **約 44.14% 減**、J は **約 43.43% 減**。教師 71,626 に対して C 39,754–41,290、J 39,754–42,314 | 16 教師ごとの比の幾何平均。品質条件で選定されたモデルであり、教師と完全に同じ精度という意味ではない |
| **L → J：同じ予算の学習方式** | NMSE は **16/16 で低下**、平均精度 **+1.76 pp**（95% CI [+1.16, +2.34]） | 同じ初期値・600 updates での training-regime 比較。教師強制・勾配経路等も変わり、目的関数だけの因果効果ではない |
| **J → C：置換部分** | 最初に品質合格した候補予算は **約 21.2% 減**。C/J = 0.788324、95% CI [0.694450, 0.904368] | 調べた有限グリッド上の値。11 教師で小、4 同じ、1 大。理論最小ではない |
| **J → C：モデル全体** | 追加削減はパラメータ **約 1.26%**、NPZ bytes **約 1.53–1.56%** | 上の教師比の削減とは別。保存形式の bytes は RAM / VRAM ではない |
| **J → C：CPU forward** | batch 1 **約 16.33%**、batch 128 **約 5.67%** 短縮 | 共有 CPU・1 thread・固定 synthetic 入力。ロード込みや GPU の保証ではない |
| **CIFAR-10：品質条件** | 元 Comp の official test 精度低下 **4.26 / 4.22 / 4.38 pp** | 許容 2 pp を超過。調べた改善・診断系列も目標未達。方法一般の不可能性を示すものではない |

[Digits の詳細・比較条件・区間](research-checkpoints/2026-10-27/README.md) · [全モデル保存量](research/public_checkpoint_20261027/research/selected_whole_storage/REPORT_JA.md) · [CPU 条件](research/public_checkpoint_20261027/research/selected_latency/REPORT_JA.md)

**教師からの圧縮効果と、強い比較対象 J に対する C の追加利得は別です。**「全体がほとんど縮まない」だけでも、「全モデルが 21% 縮む」だけでも、この比較を正しく表しません。

## 系列別の入口

| 系列 | 調べたこと・入口 |
|---|---|
| 基礎比較・学習中の統合 | [旧 Digits 基礎比較](docs/CORE_DISCOVERY_REPLICATION_DIGITS.md)、[学習中の統合](docs/TRAINING_TIME_CONSOLIDATION.md)、[追加対照](docs/LATE_STAGE_FINDINGS.md) |
| 精度・構造・保存形式 | [訂正済み precision 系列](docs/phase2/README.md)、[v10–v25 の原プロトコルと結果](archives/README.md)。置換コード、全パラメータ、実ファイル bytes を区別 |
| 画像・系列・言語モデル | [SmallViT 比較](docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md)、[歴史的拡張とブランチ別証拠](docs/RESEARCH_INDEX.md)。置換直後と追加適応後を区別 |
| 再帰合成・補正部分空間・分布変化 | [C-series と QR 診断](docs/RESEARCH_INDEX.md)。小さい補正部分空間を全モデルの圧縮率と扱わない |
| 回帰・外部妥当性 | [回帰系列の証拠](docs/RESEARCH_INDEX.md)。教師の品質・データ分割・確認条件を別途確認 |
| 実行方式・systems | [CPU PoC](docs/RUNTIME_POC.md)、[S1–S7](docs/RESEARCH_INDEX.md)。アプリが管理する arena とプロセス全体メモリを区別 |

### 再現性も別の研究結果

[R87 の記録済み新レシピ・確認実験](https://github.com/Unjuno/canaria-neural-simplification/blob/56a8d1a50f15cad0d930cc899336071fa8efe7dc/results/reproduction/r87_resolution/RESULT.md)は、検査した Intel / AMD 環境での数値整合と、別の 16 model-seed コホートでの composed/component-wise 予算比 0.531610 を報告しています。これは現在の C/J 比較とも、旧 8 seed の厳密再現とも異なります。旧実行の来歴・数値は未回復であり、[旧再現不一致](docs/REPRODUCTION_DISCREPANCY.md)を解消済みにはしません。

この公開スナップショットは Phase2E の `INVALIDATED_IMPLEMENTATION_BUG` 記録と訂正履歴を保持しています。無効化された実行を科学的証拠として扱いません。

## 確認できること、まだ言えないこと

条件付きの圧縮・近似・速度の利点はあります。一方、Digits 以外にも California Housing やモデル系列を変えた検証はありますが、それぞれの固定データ分割・比較条件を越えた一般化、未知の大規模 architecture への普遍的保証、一般的な実配備の費用削減は確立していません。隠れ状態誤差の改善とタスク品質の改善も別です。[失敗・訂正](docs/NEGATIVE_RESULTS.md)を成功結果と一緒に読む構成にしています。

## 読む・再計算する・再現する

- [保存数表の再計算](research/public_saved_analysis_20261027/README.md)、[打ち切りを含むしきい値感度](research/threshold_robustness_20261027/README.md)
- 公開 source-only パッケージ：[matched](research/digits_matched_source_20261027/README.md) / [grid](research/digits_joint_budget_source_20261027/README.md) / [storage](research/selected_whole_storage_source_20261027/README.md) / [latency](research/selected_latency_source_20261027/README.md)。不足入力を各資料で確認してください。
- [旧 baseline の Quickstart](QUICKSTART.md)、[再現性の範囲](docs/REPRODUCIBILITY.md)、[文書一覧](docs/README.md)

このページは記録済み研究の整理です。コード検査、保存数表の再計算、新しい技術的再実行、独立した科学的再現は同じ判定ではありません。原プロトコル・失敗・訂正履歴・ブランチ上の証拠を保持します。
