# Evidence Matrix — 主張と根拠フォルダ

| 主張 | 強さ | 主な証拠 | 注意 |
|---|---|---|---|
| function-fitがweight SVDより強い | strong small-scale | `canary_core_miniexperiment`, `canary_core_replacement*` | MLP/digits中心 |
| local safety != composition safety | strong small-scale | `canary_core_cnn_sequential*`, `canary_core_composition_*` | 小型CNN |
| block境界 != natural functional boundary | strong small-scale | `canary_core_span_merge_experiment`, `canary_core_auto_span_merge_*` | generalization未確認 |
| full spanをsingle spatial operatorへcompile可能 | strong small-scale | `canary_core_span_merge_experiment`, `canary_core_fullspan_*` | repair条件依存 |
| Canary != utility | strong | `canary_core_canary_diagnostic`, `canary_core_suppression_knockout_quick`, implantation/value controls | Canaryはsensor候補 |
| Adoption != utility | strong small-scale | `canary_core_mechanism_implantation*`, `canary_core_rank3_value_control*` | seed依存あり |
| iterative compilerが成立 | strong implementation evidence | `canary_core_iterative_compile_loop`, auto compile/Pareto/family dirs | 理論的最適性は未証明 |
| retrainingで有効function familyが変化 | moderate/strong | `canary_core_auto_function_family_*` | 少数seed |
| same-taskでmechanism inheritanceが高い | moderate | `canary_core_function_identity_*`, `canary_core_mechanism_inheritance_*` | small CNN |
| new taskでsuccessful mechanism reorganization | moderate | `canary_core_newtask_inheritance_*` | orientation/position tasks |
| coarse function archetypesが飽和傾向 | exploratory/moderate | `canary_core_function_atlas_gauge_combined4` | grammar finite |
| large spanのtrajectory manifoldが低次元 | moderate | `canary_core_boundary_scale_manifold_combined` | 4-block range |
| same-chart operatorを2–3 coordsで表現可能 | strong within tested chart | `canary_core_operator_coordinate_compression` | cross-seed未成立 |
| shared dictionary prototypeに情報がある | exploratory | `canary_core_large_span_dictionary_reuse` | adapterが補正を担う可能性 |
| pure cross-seed gauge alignmentでは不十分 | strong negative | `canary_core_fixed_gauge_*`, `canary_core_dictionary_boundary_followup` | canonicalization未試験 |
| composition complexityが頻繁にsubadditive | moderate | `canary_core_general_composition_law` | digits/4-block |
| large-span simplificationはadaptive redistributionが主 | strong within tested full spans | `canary_core_general_composition_law/fullspan_*` | shell capacity交絡 |
| simplificationにformation time tauがある | moderate | `canary_core_general_composition_law/adaptive_trajectory_*` | 8 conditions |
| Canaryはsimplificationのpartial sensor | exploratory/moderate | `canary_core_general_composition_law/sensor_*` | width confoundingあり |
| Canary localizes severe damaged block | strong in targeted mini tests | `canary_core_canary_weighted_kd_alltarget_summary` | artificial damage |
| Canary-weighted KDが常に最良 | false/not supported | KD folders | damage/block/alpha依存 |
| distillationはmechanism complexityで説明可能 | hypothesis | KD + dictionary/manifold evidence | 直接因果未検証 |
