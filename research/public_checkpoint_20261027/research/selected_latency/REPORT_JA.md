> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# 凍結済み選択モデルの CPU 定常 forward 時間

実行者: prepare_selected_latency_benchmark。自己実行結果と独立した保存証拠監査を区別して記録。実行は1回、再試行なし。

- Protocol commit: `5161f83`。実行 commit: `d653a73406525b42c19c7266046f3cfa0317e56c`。
- 終了: supervisor exit0 / timeoutなし / 24.935秒。全96 seed×batch×arm COMPLETE、2880 block記録。
- Exact48 whole-model states: 既存storage資料の16teacher/32最小合格C/Jを変更せずロード。既存270validation logitsは全48で最大絶対誤差0.0。新データ評価ではない。
- 同一固定synthetic入力、CPU1thread、eval/no_grad、MKLDNN無効、AVX2/MKL COMPATIBLE。batch1/128、warmup20、30round×20forward。各seed/batch内は全6順序を5回ずつ交互実行。ロード・検証・保存時間はタイミング外。

## 結果（seed内30roundの中央値、その16seedの幾何平均比）

| Batch | C/J 時間比 | C/教師 時間比 | J/教師 時間比 | C<J |
|---|---:|---:|---:|---:|
| 1 | 0.836712 | 0.698418 | 0.834718 | 16/16 |
| 128 | 0.943339 | 0.587542 | 0.622833 | 16/16 |

CはJよりbatch1で16.33%、batch128で5.67%短い。同じモデルのファイル容量差を速度に読み替えたものではなく、ここで初めて実測した条件付き結果。

## 実時間の範囲（各seedの中央値、microseconds / forward）

| Batch | Teacher min–max | C min–max | J min–max |
|---|---:|---:|---:|
| 1 | 93.246–111.070 | 65.021–75.587 | 78.098–93.360 |
| 128 | 780.942–871.653 | 460.049–521.164 | 481.125–555.584 |

## 制約・未完了

- この共有ホスト・固定synthetic入力・eager CPUのみ。CPU負荷の競合やseed順の時間変動を排除した専用機測定ではない。他ハードウェア、実入力分布、学習速度への一般化は未検証。
- 実行前loadavg: [0.0849609375, 0.03955078125, 0.00537109375]、実行後: [0.42529296875, 0.11962890625, 0.0322265625]。CPU/affinity/process censusはENV資料参照。
- 元450testの指標、外部UCI、新しい学習・選択・量子化・compile・カーネル調整は実施していない。
- 事後のばらつき記述: 177/2880 blocksがinclusive-quartile Tukey外れ値。最大block/同cell中央値比25.235倍（seed3,batch1,C）。全blockを残し、事後除外・再試行なし。共有ホストのばらつきを含む中央値比較であり、安定専用機の保証ではない。
- 4GiBはaddress-space制限であって実測RAMではない。1thread等はsourceの設定/assertionで確認し、独立したlive thread counter保存はない。

## 証拠

- `results/RAW.jsonl`: 全2880 timing blocks。total_ns、iterations、ns_per_forward、seed、batch、arm、round、position。
- `results/SUMMARY.json`: 全32 paired seed×batch行と全16集約。`results/CORRECTNESS.json`: graph/reference検証。
- `results/INPUT.npy`: timing入力。`SOURCE_MANIFEST.json`: savedvalidation/referenceの事前hash。
- `attempt/START.json`, `EXIT.json`, `SUPERVISOR_EXIT.json`, `CENSUS.json`: 一度限りの実行・fullcensus。
- synthetic controls9/9 PASS（実モデルtimingではない）。setup失敗、aggregate cap、native hang終了の回帰テストを含む。
