# C21 confirmatory report

**Status: CONFIRMATORY PASS.**

C21 tested an 8/32-dimensional teacher correction plus complementary 24D Canaria self-anchor on the established SmallViT central two-block regime. Fresh seeds were 1530–1541; 11/12 met the locked teacher eligibility threshold (>=0.95), exceeding the minimum of 8.

Worst-basis anchored-8 improved frozen in 11/11 eligible seeds. Mean D_worst was -0.00592118, 95% CI [-0.00676274, -0.00502669]. Worst-basis/full-32 geometric NMSE ratio was 1.096240x, 95% CI [1.083808, 1.108921]. Basis sensitivity was 1.014459x, 95% CI [1.010369, 1.018853].

Worst-basis validation accuracy minus full-32 averaged -2.559 pp, 95% CI [-3.333, -1.582] pp. Held-out test difference averaged -2.492 pp, 95% CI [-3.670, -1.448] pp. Both locked -4 pp safeguards passed. Test evaluation occurred only after all fitting and validation metrics within each seed were complete.

Self-anchoring also beat naive 8D sketch-only supervision in 44/44 same-basis seed pairs.

## Interpretation

Within this SmallViT-family central-two-block testbed, quarter-dimensional teacher correction plus a complementary Canaria self-anchor robustly repairs the recursive boundary across fresh model seeds and fresh identity/random bases, with bounded loss versus full hidden alignment. This does not establish full-model Transformer compression, arbitrary subspaces or architectures, LLM behavior, or a universal 8D interface.
