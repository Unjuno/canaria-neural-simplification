# Canaria — カナリア

**タスクに条件付けたニューラル計算の簡略化を調べる研究リポジトリ。**

[English](README.md) · [主張と証拠](docs/CLAIMS_AND_EVIDENCE.md) · [告知準備](docs/ANNOUNCEMENT_READINESS.md) · [研究索引](docs/RESEARCH_INDEX.md)

Canaria は、学習済みニューラルネットワークの一部を置換するとき、実装ブロックごとに別々に近似するよりも、対象span全体を一つの入出力関数として近似した方が、小さい学習済み置換計算でタスク性能を維持できる場合があるかを調べています。

**限定された研究プレビューであり、普遍的な圧縮法や実運用向け推論ライブラリではありません。** 現在の再審済み見出し証拠は、Residual-MLP・sklearn digits・最初の2ブロックという固定条件で行ったfresh direct baselineです。California Housingは、同じアーキテクチャ族における限定的なtask/dataset外的妥当性の補助証拠です。

## 現在の再審済みdirect baseline: R87R2

事前固定した `R87R2_FRESH_SQRT_PROFILE_CONFIRMATION` と、テストした `portable_v2_sqrt64` CPU数値profileの下で、fresh model seed `871200`–`871215` の16/16がeligibleでした。

- 合成方式 / component-wise方式の選択置換予算の幾何平均比: **0.5316**
- mean `log2(B_composed/B_componentwise)` のpaired bootstrap 95% CI: **[-1.0413, -0.7500]**
- 16 seed中 **15 seedで合成方式が厳密に小さい予算**、1 seedはtie
- 選択endpointのtest accuracy差（composed − component-wise）の95% CI: **[+0.00028, +0.00750]**
- locked decision: **`R87R2_CONFIRMATORY_PASS`**

ここで「選択予算」は固定された有限gridで最初に合格した点であり、数学的な最小値ではありません。16観測は同じdigits split上のmodel seedであり、16個の独立datasetではありません。test metricは置換予算の選択には使っていませんが、test split自体は既存の再利用splitです。また、この数値profileは限定されたCPU研究設定であり、CPU/GPU一般のportability theoremではありません。

[Protocol](results/reproduction/r87r2_fresh_sqrt_profile/PROTOCOL.json) · [fresh raw rows](results/reproduction/r87r2_fresh_sqrt_profile/FRESH_ROWS.json) · [Decision](results/reproduction/r87r2_fresh_sqrt_profile/DECISION.json) · [Candidate audit](tools/audit_publication_candidate.py)

## 過去の1200–1207 archiveは残すが、書き換えない

旧digits cohortの保存記録では、component-wise平均3,584、composed平均1,728で、記録上8/8 seedでcomposed側が小さいという同方向の結果でした。一方、現在の完全cohort再実行では、保存された全seed endpoint/statisticを厳密には再現できていません。[historical reproduction discrepancy](docs/REPRODUCTION_DISCREPANCY.md)を参照してください。

R87R2は、**新しい限定数値profileで取得したfresh confirmatory cohort**です。旧1200–1207値の復元ではありません。Issue #87はhistorical reproduction/provenance debtとしてopenのまま残し、旧保存値をR87R2の値で上書きしません。

## 限定的な回帰補助証拠: California Housing Phase4

固定した1つのCalifornia Housing split上で、同じResidual-MLP族を使い、Stage Aでheld-out testを使わずにteacher recipeを事前選択した後、fresh 8-seed Stage Bは `PHASE4_CONFIRMATORY_PASS` でした。

- 8/8 seedでcomposed側が厳密に小さい選択置換予算
- composed/component-wise選択予算の幾何平均比: **0.4760**
- mean log2 budget ratioのpaired bootstrap 95% CI: **[-1.3282, -0.8325]**
- 選択held-out-test R²差（composed − component-wise）の95% CI: **[+0.00436, +0.00967]**
- teacher held-out-test R²平均の95% CI: **[0.7911, 0.7996]**
- 第2 hosted workerで8 scientific recordをexact technical replication

これは1 dataset・1 splitです。task/dataset外的妥当性を限定的に補強しますが、architecture universalityは示しません。選択予算も固定有限grid内の最初の合格点です。

[Stage-B protocol](results/phase4/california_regression/STAGE_B_CONFIRMATORY_PROTOCOL.json) · [fresh raw rows](results/phase4/california_regression/stage_b_confirmatory/FRESH_ROWS.json) · [Decision](results/phase4/california_regression/stage_b_confirmatory/DECISION.json)

## 証拠classを分けて監査する

再審済みpublication candidateは、NumPy 2.3.5を入れた環境で次を実行して監査できます。

```bash
python tools/audit_publication.py
python tools/audit_publication_candidate.py
```

後者はR87R2とPhase4について、vendorしたraw rowsからlocked aggregate decisionを再計算し、元research commitで再審したGit blobと内容が同一であることも検査します。

旧1200–1207 cohortの厳格再現は別経路として残しています。

```bash
python tools/audit_publication.py --require-reproduction
```

Issue #87が未解決の間、このhistorical strict gateはBLOCKEDのままです。R87R2を使って旧archiveをPASS扱いすることはありません。[QUICKSTART.md](QUICKSTART.md)はhistorical core rerunの手順を残しています。

## 支持しない主張

現時点で支持するのは、明示したprotocol・span・replacement familyにおける限定的なcomposition-budget効果です。普遍的complexity law、whole-model compression、LLM一般への適用、普遍的minimum interface dimension、runtime speedup、RAM/VRAM削減、energy削減、hardware全般へのportabilityは主張しません。

[SmallViT direct-composition evidence](docs/CROSS_FAMILY_COMPOSITION_REPLICATION.md)は、digits上の限定two-block spanという境界付きのsupporting architecture-family evidenceです。Systems S1–S7はprototype測定に限定します。Imported C59/C60はResidual-CNN系列であり、original C61は未解決です。別architectureのResidual-MLP C61R系列でoriginal C61を修復したとは扱いません。

否定的結果と無効化も残します。Phase3Bは事前登録teacher-strength gateがuncertainだったためpositive stronger-teacher headlineから除外します。Phase3Cはboundary evidenceとして保持します。Phase2Eは **`INVALIDATED_IMPLEMENTATION_BUG` / `DO_NOT_USE_FOR_INFERENCE`** のままです。Phase2Iのcausal attributionは撤回済み、Phase2Oはrepair-sample advantageを確立していません。[否定的結果](docs/NEGATIVE_RESULTS.md)

## 公開状態

機械可読な現行選択は[publication/CLAIM_POLICY.json](publication/CLAIM_POLICY.json)、post-v0.2独立再審の採否は[publication/POST_V02_CLAIM_LEDGER.json](publication/POST_V02_CLAIM_LEDGER.json)にあります。merge、release tag、announcement、Issue #13 close、external reproduction、peer reviewはそれぞれ別イベントであり、CI PASSだけでそれらを宣言しません。

現在のgateは[STATUS.md](STATUS.md)と[告知準備](docs/ANNOUNCEMENT_READINESS.md)を参照してください。凍結tag `v0.2.0-public-snapshot`はhistorical provenanceであり、現在のready certificateではありません。

引用時は正確なcommit・protocol・resultを指定してください。ライセンスは[Apache-2.0](LICENSE)です。
