# カナディア研究 引き継ぎパッケージ v10 — START HERE

このZIPは、これまでのカナディア／Canary研究を次のチャット・研究者・エージェントへ引き継ぐためのアーカイブです。v9に続き、v10でrecursive recompile・post-shell capacity frontier・boundary-local compile・recursive repair pilotを追加しました。

## まず読む順番
1. `docs/01_RESEARCH_HANDOFF.md` — 研究全体の一本化された説明
2. `docs/02_CONFIRMED_FINDINGS.md` — 現時点で比較的強く確認できたこと
3. `docs/03_OPEN_QUESTIONS.md` — 未解決点・交絡・反証可能性
4. `docs/07_CURRENT_GENERAL_THEORY.md` — 現在の仮説階層
5. `docs/20_NUMERICAL_TRACEABILITY.md` — 主要数値のraw照合方法
6. `docs/21_CANARY_BLIND_DECISIVE_PROTOCOL.md` — **次に実施する最優先実験の確定仕様**
7. `docs/22_REPRODUCIBILITY_AND_FINAL_AUDIT.md` — 再現性限界と最終監査
8. `docs/23_V9_EXPERIMENT_PROGRESS.md` — **v9で実際に進めた実験と結果**
9. `docs/24_PHASE_A_PRECONFIRMATORY_AMENDMENT.md` — **confirmatory開始前の固定変更**
10. `docs/25_V10_RECURSIVE_RECOMPILE_RESULTS.md` — **recursive compiler仮説の追加pilotと否定結果**
11. `indices/claim_registry.csv` — 主張→根拠→scope/caveat
12. `indices/experiment_catalog.md` — 全実験フォルダ一覧
13. `raw_experiments/` / `scripts/` — 元出力・再実行候補

## 研究の現在地（一文）
**Canaryは根本現象ではなく、ニューラルネット全体に存在する可能性のある「合成による適応的簡約」を境界から部分観測するsensorであり、boundary expansionによって分散計算が少数の低次元・再利用可能な実効機構へcollapseする、という一般則候補を検証している。**

## 最重要の現在仮説
候補となる上流則は、一定のrepair budget `τ` のもとでの経験的subadditivityである。

`C(A∪B;τ) < C(A;τ) + C(B;τ)`

ただし、これは現時点では小型digits CNN中心の経験則候補であり、普遍定理ではない。

## v5→v8で追加した重要点
- v4 checksumの1件不一致を検出し、最終manifestを再構築。
- 主要27数値をraw CSVから再照合し、27/27を整合確認。
- 主張registryとexperiment→script provenance mapを追加。
- Canary-blind Phase AをH/T/D/C/U形式で事前規定。
- Stage-1 blind tableをhash lockしてからCanaryを測るblinding barrierを追加。
- historical environmentが未固定という再現性限界を明記。
- self-audit scriptを追加。

## 注意
- 主実験は sklearn digits と小型CNN中心。一般性は未確立。
- 多くの比率は **target core parameter/storage ratio** であり、モデル全体比ではない。
- 短期repairと長期repairで結論が変わる場合がある。
- coarse archetype saturationはfinite universal theoremではない。
- cross-seed global mechanism dictionaryは未完成。
- `current_audit_environment.json` は元実験環境ではない。
- 最終判断は `claim_registry.csv` のscope/caveatとraw evidenceを確認する。

## v9追加の要点
- plain 8-blockはpilotで学習不能だったため、confirmatory前にresidual 8-blockへ変更。
- seed1000–1007 baseline eligibilityは8/8 PASS（Canary/simplification未測定）。
- full-span repairはpost-shellのみでmatched-control utility約0.98まで回復。
- tau=0 blind map pilotではlow-Canary strong simplification 32/45=71.1%。
- ただしn=3 pilotでありconfirmatory結論ではない。

## v10追加の要点
- full-core compile後のpost-shell全体を単一linear mapへ再compileするとutility約0.75–0.76まで低下し、0/3 seedで0.95基準を満たさなかった。
- 1-hidden-layer headを幅48まで拡張してもmatched-control utility 0.95を安定達成しなかった。
- `compiled core + Conv1 b_out + linear head` は約6.45k params（baseline比約82%削減）まで縮約できるが、seed-stableな0.95 utilityは未達。
- compact modelを追加repairしてもmatched controlの改善が速く、相対utilityは低下。
- 強い「無限recursive simplification」仮説はpilotでは不支持。現在はtask-conditioned nonlinear complexity floor / recursive fixed-point候補を検証対象とする。
