# 比較的強く確認できたこと

## A. Canary / sensor
1. Canaryはutilityそのものではない。
2. low Canary != high utility。
3. strong adoption != high utility。
4. Canary suppressionだけではcompressibilityは上がらない。
5. Canaryはboundary stress / contract sensitivityとして扱うのが最も整合的。

## B. Boundary / functional structure
6. 実装block境界は自然な機能境界とは限らない。
7. 個別blockで危険でも、複数blockをmergeしたfull spanは単一の簡単なoperatorへ置換できることがある。
8. local safety != composition safety。
9. boundary expansion後のlarge spanは、一度成立すると大きなparameter圧縮を与える。

## C. Iterative compile
10. compile → freeze → retrain → recompile の反復は実装上成立。
11. 自動span/function-family selectionとautomatic stopも小規模実験では成立。
12. retrainingにより後段で有効なfunction familyが変わることがある（例: depthwiseが後から有効）。

## D. Mechanism identity
13. same-task retrainingでは主要operator basisが高率で継承される。
14. task shiftではutility獲得と同時にinheritance低下・new residual growthが起こる。
15. 「毎回完全に新しい関数」より「inheritance + residual absorption」が良いモデル。

## E. Function families / atlas
16. 実装名とbehavioral mechanism typeは一致しない。
17. coarse resolutionでは少数archetypeへ飽和する兆候。
18. 細かいparameter variationは連続manifoldとして残る。

## F. Large-span low dimensionality
19. span width増加とmechanism trajectory dimension低下の相関が観測された。
20. full-span operator trajectoryは2〜3 coordinatesで高精度再構成できるケースがある。
21. dictionary + coordinatesによるstorage amortizationの可能性がある。

## G. General composition law
22. 4-seed 80 composition eventで72.5%が記述複雑度のsubadditive simplification。
23. width増加でadaptive simplification頻度が上がる傾向。
24. 簡約parentの内部にもsimplificationが再帰的に存在することが多い。
25. full spanではintrinsic fitよりadaptive recontractingが簡約成立の主要因となるケースがある。
26. simplificationには形成時間 tau がある。
27. Canaryはsimplificationについて情報を持つが、lawそのものではない。
