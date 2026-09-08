# 研究時系列（概念上のフェーズ）

> 実ファイルのmtimeは再生成・コピーで変わり得るため、ここでは会話・実験の論理順を採用する。

## Phase 0 — 初期圧縮・Canary診断
- MLP/CNN低rank置換
- SVD vs hidden fit vs function fit
- replacement controls / null controls
- Canary metric diagnostic
- threshold frontier / forward canary selection
- 結果：Canaryは候補安全性の一部を捉えるが、単独のutility指標ではない。

代表フォルダ：
- `canary_core_miniexperiment`
- `canary_core_replacement*`
- `canary_core_cnn_*`
- `canary_core_canary_diagnostic`
- `canary_core_cnn_threshold_frontier`

## Phase 1 — Composition / stitching / local-to-global
- 個別block replacement
- sequential replacement
- stitching/folding
- composition-aware greedy / rollback
- 結果：local safety != composition safety、複数spanのfoldが可能。

代表：
- `canary_core_cnn_sequential*`
- `canary_core_stitching_experiment`
- `canary_core_composition_*`

## Phase 2 — Function-family probing
- linear 1x1/3x3/5x5/7x7
- low-rank
- depthwise+pointwise
- bottleneck nonlinear
- affine / constant 等
- 結果：観測manifold上ではspatial linear/low-rankが強いが、固定familyではない。

代表：
- `canary_core_operation_probe`
- `canary_core_operation_family_map_quick`
- `canary_core_function_family_all_replace`
- `canary_core_compression_family_frontier`

## Phase 3 — Boundary expansion / span merge
- implementation blockを越えてspanをmerge
- full-span single-function replacement
- 結果：自然な機能境界は実装境界より大きいことがある。

代表：
- `canary_core_span_merge_experiment`
- `canary_core_auto_span_merge_*`
- `canary_core_span_auto_policy_existing`

## Phase 4 — Automatic iterative compiler
- fixed threshold → utility trial → Pareto knee → 4-objective Pareto
- function familyも探索対象へ
- expression grammar synthesis
- 結果：compile→freeze→retrain→recompile→stopを自動化。

代表：
- `canary_core_auto_compile_*`
- `canary_core_auto_risk_from_utility*`
- `canary_core_auto_pareto_knee_*`
- `canary_core_auto_multiobjective_*`
- `canary_core_auto_function_family_*`
- `canary_core_expression_synthesis_*`
- `canary_core_iterative_compile_loop`

## Phase 5 — Mechanism implantation / adoption / utility
- extracted mechanismをbranchとして埋め込む
- adoption ablation
- rank3 value controls
- trajectory selector
- 結果：Adoption != Utility。強く使われても有害な機構がある。

代表：
- `canary_core_mechanism_implantation*`
- `canary_core_latent_component_implant*`
- `canary_core_rank3_value_control*`
- `canary_core_trajectory_selector`

## Phase 6 — Identity shift / inheritance / new mechanisms
- compile前後のoperator/Jacobian/subspace比較
- same-task long trajectory
- task shift/new-task
- 結果：same-taskはinheritance高、新taskでは成功適応しながらidentityが崩れる。

代表：
- `canary_core_function_identity_*`
- `canary_core_mechanism_inheritance_*`
- `canary_core_taskshift_inheritance_*`
- `canary_core_newtask_inheritance_*`

## Phase 7 — Function atlas / archetype saturation
- seed/span/stateをまたいで置換可能functionを収集
- raw channel座標の問題を発見
- gauge-invariant fingerprintへ移行
- 結果：coarse archetypeは少数へ飽和する兆候、細部はcontinuous manifold。

代表：
- `canary_core_function_atlas_*`
- `canary_core_fullspan_atlas_trial4_combined`
- `canary_core_fullspan_archetype_dimension`

## Phase 8 — Dictionary / low-dimensional coordinates
- cross-context prototype reuse
- operator PCA coordinates
- boundary widthとmanifold dimension
- cross-seed gauge alignment
- 結果：same-chartでは2〜3 coordinatesが強い。global dictionaryはalignmentで未解決。

代表：
- `canary_core_large_span_dictionary_*`
- `canary_core_operator_coordinate_compression`
- `canary_core_mechanism_dictionary_coordinates`
- `canary_core_fixed_gauge_*`
- `canary_core_boundary_scale_manifold_*`
- `canary_core_dictionary_boundary_followup`

## Phase 9 — Distillation / repair接続
- damaged K blockをCanaryで検出
- CE / feature KD / logit KD / Canary-weighted KD
- assembly repair
- 結果：Canaryはdamage localizationに強いケースがあり、Canary-weighted KDは一部damageで有利だが一貫してCE/KDを上回るわけではない。

代表：
- `canary_core_canary_weighted_kd_*`
- `canary_core_ce_kd_*`
- `canary_core_assembly_kd_repair_mini`
- `canary_core_adaptive_alpha_policy`

## Phase 10 — General composition / simplification law
- C(S)を最小明示記述量として定義
- G(A,B)でsubadditivity計測
- intrinsic vs adaptive simplification分離
- simplification formation time tau
- Canary sensor predictive test
- 結果：4-seed/80 eventsで72.5%簡約、large spanはadaptive collapseが支配的。

代表：
- `canary_core_general_composition_law`

## 現在
**次フェーズはCanaryを探索選択に使わないblind/exhaustive simplification map。**
これによりCanary-local theory / Canary-observer theory / general composition theoryを直接比較する。
