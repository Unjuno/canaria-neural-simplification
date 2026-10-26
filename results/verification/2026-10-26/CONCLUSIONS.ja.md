# Canaria：現時点で成立していること

2026年10月26日確認。対象は固定済み候補 PR101（`056d69c`）と SmallViT の PR105（`50a73cd`）、および同日のローカル検証記録です。追加証拠はローカルの証拠ブランチ `142287214260a1c9b87d5566c9fb0084a0806a82` に保存済みで、push は行われていません。

## 結論

**「ブロックごとに置き換えるより、複数ブロックをまとめて置き換える方が、所定の精度条件を満たす置換パラメータ数を小さくできる」という限定された実験結果は成立しています。** digits の R87R2 と California Housing の Phase4 は、保存結果の読み直しだけでなく、固定した条件での実学習から結果まで再現されました。

ただし、これは **モデル全体の半減、推論速度・メモリの改善、一般的な半減則、LLM への適用の証明ではありません。** 比較しているのは所定の候補グリッド内で選ばれた「置換部分」の予算です。[主張と適用範囲の固定記録](https://github.com/Unjuno/canaria-neural-simplification/blob/056d69cc257a7670fb45e17f11588afa497d8881/docs/CLAIMS_AND_EVIDENCE.md)

## 証拠の強さを分ける

| 対象 | 確認できたこと | ここからは言えないこと |
|---|---|---|
| **R87R2 / digits** | 既知 seed の bridge と固定16 seeds、計17プロセスが正常終了。実学習の科学的 outcome 全体と判定が保存値に完全一致。別実装による保存値監査も一致。予算比の幾何平均 **0.532**、15/16 seeds で小さく、事前指定の精度非劣性条件を通過。 | 新しい独立 seeds・独立データによる再確認ではない。古い8-seed記録の不一致や Issue #87 を解決しない。全 trace 配列のホスト間一致、外部研究チームによる追試も意味しない。 |
| **Phase4 / California Housing** | 保存値の再計算978 checks に加え、**固定8 seeds の実学習が完了し、全科学的記録と判定が保存値に完全一致**。独立に実装した監査も154/154 checks を通過し、62個の封印対象ファイル、データ・分割ハッシュ、bootstrap を確認。8/8 seeds で小さく、予算比 **0.476**、選択後 test R² の非劣性条件を通過。 | 同じ8 seeds の技術的再実行であり、独立サンプル数は増えない。実学習には固定済みの完全な Phase4 ソースを使用。PR101 の基底 runner 欠落はローカルで補いレビュー済みだが、**修正候補からの全学習再実行は実施していない。** |
| **SmallViT / hidden-test** | 保存記録の評価器と closure audit がともに完全一致。16試行中13適格 seeds、全13で小さい予算を選択し、幾何平均比 **0.512**。コードの選択／test評価順序と変異チェックも通過。 | 再学習ではない。test データは同一プロセス内にあり、過去の実際のアクセス隔離を証明しない。compiler holdout は teacher 学習データの部分集合。 |
| **SmallViT / objective control** | 保存 logits などの整合性と再計算を確認。15適格 seeds の判定は **UNCERTAIN**。test 非劣性は通るが、NMSE 改善条件は成立していない。 | 「共同最適化が改善を生む」という機構の確証としては使えない。効果がないことの証明でもない。 |

固定された元結果：[R87R2](https://github.com/Unjuno/canaria-neural-simplification/blob/056d69cc257a7670fb45e17f11588afa497d8881/results/reproduction/r87r2_fresh_sqrt_profile/DECISION.json)、[Phase4](https://github.com/Unjuno/canaria-neural-simplification/blob/056d69cc257a7670fb45e17f11588afa497d8881/results/phase4/california_regression/stage_b_confirmatory/DECISION.json)、[SmallViT hidden-test](https://github.com/Unjuno/canaria-neural-simplification/blob/50a73cd44a05a62e28c94a9e67c84f7b7e9e86cf/results/replication/vit_hidden_test_confirm/DECISION.json)、[objective control](https://github.com/Unjuno/canaria-neural-simplification/blob/50a73cd44a05a62e28c94a9e67c84f7b7e9e86cf/results/replication/smallvit_objective_control/DECISION.json)。同日の追加検証はローカルのハッシュ付き監査・実行記録に基づきます。これらの GitHub リンク自体が追加検証の公開を意味するわけではありません。

### 「約半分」の重要な留保

SmallViT hidden-test は、全13 seeds の componentwise 選択が**二ブロック候補の最小グリッド**に達しています。したがって約半分という数字には、「二つの置換器」と「一つの置換器」を比べる候補構造が強く関わります。これは固定候補内の結果を取り消しませんが、真の最小サイズや普遍的な圧縮法則の証拠にはなりません。モデル seed の数を増やしても、この構造的限界や単一データ分割の限界は解消しません。

### 別系統の runtime PoC は存在する

既存の G7 seed 4300 の小型 decoder 実験では、保存 artifact が **110,093 → 54,646 bytes**、CPU batch-128 推論の平均が **47.05 → 23.11 ms** となった記録があります。これはその小型モデル・CPU条件での実行利益を示す別系統の PoC であり、今回の再実行でも R87R2／Phase4 の結果でもありません。RAM削減や一般的なGPU・LLM高速化の実証にはなりません。「資源に関する測定が一切ない」という意味ではなく、限定された既存測定を今回の主張へ転用しない、という区別です。[固定された runtime PoC 記録](https://github.com/Unjuno/canaria-neural-simplification/blob/056d69cc257a7670fb45e17f11588afa497d8881/docs/RUNTIME_POC.md)

## 仕上げに必要なもの

**限定的な research preview の科学的説明に、一般則や実用高速化の実証まで追加する必要はありません。** 必要なのは、成立した限定結果と未成立の主張を分離した、追跡可能な証拠一式です。

1. **梱包修正はローカル検証済み。公開側への反映は未実施。** 修正 `3252e1baa13241817b0e32969ab0468f38abcf0f` は、欠けていた `scripts/phase3/regression_external_validity/run_seed.py` と説明文書の2ファイル追加だけ。runner は実学習で使った固定元ソースの blob `8dd3ddefb59ba250f5deca2d98d04fbff0055748` と同一で、既存の科学的コード・閾値・結果は変更していない。別レビューで3スクリプトのコンパイル／オフライン import を確認し、実施担当の18 tests・repository audit PASS 記録も確認済み。この技術レビューの承認範囲はローカルの依存ファイル補完であり、修正候補からの全学習再実行や公開準備完了を意味しない。研究ブランチの定期更新は既にユーザー承認済みで、同じ範囲の push に新たな承認は不要。現在のリモート反映の障害は、書込み認証と「CIを起動しない」条件の確保である。main マージ・リリース・対外的な主張の公表は別途承認が必要。
2. **再現用の入口を整える。** 固定コミット、環境、実行手順、ハッシュ付き記録と今回の監査を結びつける。既存監査には元の作業パスへの依存があるため、保存済み原本は変更せず、別環境で使えるラッパー／手順を用意する。これは新しい科学的実験ではなく、検証可能性の仕上げです。
3. **主張を限定したまま確定する。** 古い再現不一致、SmallViT のグリッド下限と objective-control の UNCERTAIN、実用性能未検証を本文から外さない。保存値監査・同一固定 cohort の技術的再現・独立追試を混同しない。

さらに強い主張をする場合にだけ、それに対応する追加証拠が不可欠です。**実用高速化／省メモリ**には同等品質・同等実行条件での実測、**候補構造を越える優位性**にはグリッド下限を切り分ける事前固定の比較、**一般化**には独立のデータ・分割・対象アーキテクチャ、**機構の断定**には未成立だった効果と介入の交絡を区別する検証が必要です。

本評価は科学的な証拠範囲の整理であり、リリース・マージの承認や外部査読の完了を意味しません。
