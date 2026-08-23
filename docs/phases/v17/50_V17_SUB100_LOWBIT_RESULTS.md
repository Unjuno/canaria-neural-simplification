# V17 sub-100-byte low-bit experiments

## Phase N — prospective sub-100-byte sweep (seeds 2100–2107)
At tau=2 full-shell repair, all five sub-100-byte candidates had mean matched-control utility >0.97. Bootstrap lower 95% CI exceeded 0.95 for all five in this exploratory cohort. Strong examples:
- kernel-block 24 x 2-bit, 80 B: U=0.9824, CI [0.9710,0.9921], PASS95 7/8.
- kernel-block 16 x 3-bit, 81 B: U=0.9853, CI [0.9738,0.9983], PASS95 8/8.
- spatial-offset 4 x 2-bit, 83.125 B: U=0.9733, CI [0.9635,0.9850], PASS95 8/8.
- 1:4 x 2-bit, 90 B: U=0.9824, CI [0.9694,0.9947], PASS95 8/8.
- kernel-block 20 x 3-bit, 94.5 B: U=0.9888, CI [0.9772,1.0009], PASS95 8/8.

## Phase O — independent holdout (seeds 2200–2207)
At tau=2, only 1:4 x 2-bit (90 B) retained a lower 95% CI >=0.95:
- 90 B: U=0.9790, CI [0.9535,1.0024], PASS95 6/8.
- 80–94.5 B alternatives had mean U ~0.970–0.976 but lower CI 0.931–0.941.
Thus Phase N overestimated stability for most candidates; the 90 B representation replicated.

## Phase P/Q/R/S — scale-sharing from 90 B to 76 B
The 90 B representation spends 16 B on eight FP16 per-output scales. Sharing scales gives:
- 1 scale: 76 B
- 2 scales: 78 B
- 4 scales: 82 B
- 8 scales: 90 B

Phase P (seeds 2300–2307) suggested 76/78 B stability at tau=2, but independent Phase Q (2400–2407) did not meet the lower-CI >=0.95 criterion at tau=2.

Exploratory Phase R on the Phase-Q cohort showed longer repair restores stability:
- tau=8, 76 B: U=0.9798, CI [0.9684,0.9919], PASS95 8/8.
- tau=8, 78 B: U=0.9869, CI [0.9750,1.0004], PASS95 8/8.

Phase S then independently confirmed tau=8 on new seeds 2500–2507:
- PRIMARY 76 B: U=0.9720, CI [0.9597,0.9865], PASS95 7/8 — PASS.
- 78 B: U=0.9835, CI [0.9657,1.0015], PASS95 8/8.
- 90 B reference: U=0.9662, CI [0.9543,0.9791], PASS95 6/8.

## Phase T/U — sharing the 1:4 support pattern
The 76 B model uses 36 B for 144 2-bit kernel values, 2 B for 8 2-bit biases, 36 B for independent per-output 1:4 pattern indices, and 2 B for one FP16 scale.

Phase T (seeds 2600–2607) shared the 1-of-4 pattern across output channels while preserving the same 152 stored values:
- independent pattern/output: 76 B, U=0.9828, CI [0.9636,1.0031].
- shared in output pairs: 58 B, U=0.9748, CI [0.9427,1.0005].
- shared in groups of four outputs: 49 B, U=0.9711, CI [0.9430,0.9962].
- one pattern shared across all 8 outputs: 44.5 B, U=0.9817, CI [0.9577,1.0098], PASS95 7/8.

Because four conditions were explored in Phase T, Phase U pre-specified only the 44.5 B condition on new seeds 2700–2707.

Phase U independent confirmation:
- PRIMARY 44.5 B: U=0.9915, CI [0.9636,1.0184], PASS95 7/8 — PASS.
- 76 B reference: U=0.9831, CI [0.9623,1.0047].

## Phase V/W — exact serialization without changing the model
The signed "2-bit" quantizer has qmax=1 and emits only {-scale,0,+scale}; it is ternary, not four-level.

Phase V encoded the confirmed 44.5 B model exactly as:
- 152 ternary coefficients: 5 trits/byte -> 31 B.
- 18 shared 1-of-4 indices: 36 bits -> 5 B.
- one FP16 scale: 2 B.
Total: exactly 38 B. Decoding produced max weight error 0 and identical predictions for all 8 Phase-U seeds.

The 152 ternary symbols were highly zero-heavy (zero fraction 0.461–0.934). Phase W used an exact enumerative codec for nonzero positions plus one sign bit per nonzero:
- serialized bytes per seed: 38,18,28,33,28,31,34,17.
- mean: 28.375 B.
- median: 29.5 B.
- range: 17–38 B.
This is bit-exact to the confirmed 44.5 B model, so prediction/utility is unchanged by construction.

## Current experimental frontier
- Stable independently confirmed functional representation: 44.5 B nominal structured ternary core at tau=8.
- Exact fixed ternary serialization: 38 B.
- Exact zero-aware enumerative serialization: median 29.5 B, mean 28.4 B across the confirmation seeds (variable length 17–38 B).
- FP32 Conv3 reference core: 2336 B.
