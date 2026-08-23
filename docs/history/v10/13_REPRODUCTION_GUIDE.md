# 再現・再実行ガイド

## 1. まずraw resultを読む場合
各 `raw_experiments/<experiment>/` 内の優先順位：
1. `report.md` または `*_report.md`
2. `summary.csv` / `selected.csv` / `history.csv`
3. raw `results.csv` / `trial_candidates.csv`
4. PNG図

旧実験と新実験が矛盾した場合は、以下を確認：
- bug fix前/後か
- repair epochsが同じか
- parameterではなくbufferもstorage countに入っているか
- raw channel coordinatesかgauge-invariant fingerprintか
- matched controlの学習時間が同じか

## 2. スクリプト
`scripts/` に研究実行スクリプトを収録。ファイル名は主に `run_canary_<topic>.py`。

実験結果は多くの場合 `/mnt/data/canary_core_<topic>/` に出力する設計。別環境で再実行する場合はoutput pathを検索・変更すること。

## 3. 主要依存
スクリプトのimport scanは `indices/python_imports.csv`。
典型的には：
- Python 3
- numpy
- pandas
- torch
- sklearn
- matplotlib
- scipy（解析により使用）

バージョン固定ファイルは過去実験で一貫して保存されていないため、完全bitwise reproductionは保証しない。次フェーズではrequirements/seed/hardware metadataを必ず保存する。

## 4. Seed
多くの比較はseed0〜3。seedの意味は各scriptを確認。torch/numpy/randomの全seedが固定されているかは旧scriptごとに差があるため要監査。

## 5. 主要な再現候補
### A. Iterative compile
`run_canary_iterative_compile_loop.py`

### B. Automatic family compiler
`run_canary_auto_function_family.py`

### C. Function identity / inheritance
`run_canary_function_identity_shift.py`
`run_canary_mechanism_inheritance_growth.py`
`run_canary_newtask_inheritance.py`

### D. Function atlas
`run_canary_function_atlas_saturation.py`
`run_canary_function_atlas_gauge.py`

### E. Boundary-scale manifold
`run_canary_boundary_scale_manifold.py`

### F. General composition law
`run_canary_general_composition_law.py`

## 6. 次回から必須保存するrun metadata
各runに `run_metadata.json` を追加し以下を保存：
- git/script SHA256
- Python version
- torch/numpy/sklearn versions
- CPU/GPU
- all seeds
- dataset split hash
- train/repair epochs
- optimizer/lr
- candidate grammar version
- complexity definition version
- Canary sensor definition version
- timestamp

## 7. 重要な再現性ルール
- selection用validationと最終testを分離する。
- candidateを比較する際はmatched no-compile controlに同じ追加epochを与える。
- adaptive simplificationではtauを明記する。
- full-span fit errorとpost-repair utilityを混同しない。
- `core ratio` と `full model ratio` を別列で保存する。


## 8. v8追加の再現性資材
- `environment/REPRODUCIBILITY_LIMITS.md`
- `environment/current_audit_environment.json`
- `environment/requirements_inferred.txt`
- `schemas/run_metadata_schema.json`
- `indices/experiment_provenance_map.csv`
- `scripts/audit_handoff.py`

historical package versionsは復元できないため、current audit environmentを過去runのenvironmentとして扱わない。
