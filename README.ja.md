# Canaria — カナリア

**タスクに条件付けたニューラル計算の簡略化を調べる研究リポジトリ。**

[English](README.md) · [再現手順](QUICKSTART.md) · [主張と証拠](docs/CLAIMS_AND_EVIDENCE.md) · [研究索引](docs/RESEARCH_INDEX.md)

学習済みネットワークの一部を置換するとき、実装ブロックごとに近似するより、合成した入出力関数として近似した方が、小さい置換モデルでタスク性能を保てる場合があるかを調べています。

**研究プレビューです。普遍的な圧縮法や、実運用向けの推論ライブラリではありません。**

> **告知前の再現確認で不一致を検出しました。** 2環境の全8 seed再実行で合成方式が小さい傾向は維持されましたが、一部の選択予算と精度は過去の保存値に一致しません。[再現不一致](docs/REPRODUCTION_DISCREPANCY.md)を参照してください。厳密再現と告知準備の完了は宣言していません。

## 過去に記録された結果

Residual-MLP・digits・最初の2ブロックという固定条件で、8つのモデルseedすべてで、合成方式の選択された置換パラメータ数が少なくなりました。seed 1200では個別方式3,072、合成方式1,536。8 seed平均はそれぞれ3,584と1,728です。「最小」は試した候補グリッド内の合格点を指し、理論的最小値ではありません。

[事前プロトコル](results/core_discovery_digits/PROTOCOL_LOCK.json)と[保存済み結果](results/core_discovery_digits/confirm_summary.json)に照合できます。同じデータ分割上の8モデルであり、8データセットの検証ではありません。再現手順はPython 3.11・CPU環境を固定した[QUICKSTART.md](QUICKSTART.md)にあります。

## 誤解してはいけないこと

置換パラメータ数、補正部分空間の次元、保存バイト数、推論時間、実メモリは別の指標です。いずれかの削減から他の削減は導けません。最近の9次元補正の確認実験も、全モデルを64から9へ圧縮した結果ではありません。

最新系列は[研究索引](docs/RESEARCH_INDEX.md)から元commitのコードと結果へ進めます。確認実験・探索・事後診断・原資料未回収を区別し、mainの見出し成果に自動昇格させません。元C59/C60のResidual CNNと、C61R以降のResidual-MLPは別系列です。

失敗・不確実・無効化も保存しています。Phase 2Eは`INVALIDATED_IMPLEMENTATION_BUG`で推論に使えません。Phase 2Oは修復サンプル数の優位性を確立していません。[訂正と否定的結果](docs/NEGATIVE_RESULTS.md)

## 状態

[STATUS.md](STATUS.md)と[告知チェック](docs/ANNOUNCEMENT_READINESS.md)が現行の公開範囲を定義します。凍結タグ`v0.2.0-public-snapshot`や過去のレビュー完了は、現在の全研究の完成・独立第三者再現を意味しません。

[履歴](archives/README.md)は内容を変えずに保存し、引用時は正確なcommitと使用したプロトコル・結果を指定してください。ライセンスは[Apache-2.0](LICENSE)です。
