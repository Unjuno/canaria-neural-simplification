# 現在の一般理論候補

## 仮説階層

### 上流現象
**Adaptive computational simplification under composition**

NN全体で、ある部分計算をまとめ、境界を固定し、一定時間recontractingを許すと、適応後の最小実効記述複雑度が劣加法的に減ることがある。

\[
\mathcal C(A\cup B;\tau)<\mathcal C(A;\tau)+\mathcal C(B;\tau)
\]

### 中間現象
**Mechanism collapse / residual absorption / inheritance**

合成後は少数archetype・低次元chartへcollapseし、same-taskでは旧機構を継承、新taskではresidualが増え、再compile時に吸収される可能性。

### 観測系
**Canary / boundary stress**

Canaryはこの上流過程が境界contractへ与えるstressを部分観測する。Canaryが強い場所は「簡約可能性が見えやすい」可能性があるが、simplificationはCanaryのない場所にも存在する可能性がある。

## 今後の最重要判定
Canaryを一切使わないblind simplification mapを作り、Canary mapと独立比較する。

### H1 Canary-local
simplificationはほぼCanary強領域のみ。

### H2 Canary-observer
simplificationは全域にあるがCanary強領域で観測しやすい。

### H3 General composition law
Canaryとは独立に、NN全域でstatistical subadditivityが成立する。

現在の本命はH2〜H3。ただし決定的実験は未実施。
