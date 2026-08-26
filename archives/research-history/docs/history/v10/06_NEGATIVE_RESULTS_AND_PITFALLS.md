# Negative Results / Pitfalls

1. Canary threshold aloneをsafety ruleにすると有効候補を落とす。
2. Canary suppressionはcompressibility改善を保証しない。
3. 強いbranch adoptionはutility改善を保証しない。
4. early epoch utility signは不安定。
5. 単純なhandwritten composite scoreは小標本で失敗。
6. 個別blockの安全性を合成してもglobal safetyは保証されない。
7. depthwise direct fitが悪くてもrecontracting後には有効になり得る。
8. short repairはlarge span viabilityを大幅に過小評価する。
9. raw channel coordinatesでcross-seed clusteringするとbasis permutation/rotationを別機構と誤認する。
10. pure 1x1/3x3 linear gauge alignmentだけではcross-seed global dictionary共有は十分でなかった。
11. trainable adapterが成功しても、gauge alignmentだけでなくmissing mechanismを補っている可能性がある。
12. finite candidate grammarからの飽和は、数学的finite mechanism theoremではない。
13. full-span small operatorはdirectには性能が壊れ、recontract後に成立することがある。これを「元写像が単純」と誤解しない。
14. bufferをparameter countから除外すると記述量0などのバグが起きる。storage complexityではbufferも数える。
