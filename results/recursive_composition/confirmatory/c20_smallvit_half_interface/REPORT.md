# C20 confirmatory report

**Status: CONFIRMATORY PASS.**

C20 tested a 16/32-dimensional teacher correction plus complementary 16D Canaria self-anchor on the established SmallViT central two-block recursive-replacement regime. Fresh model seeds were 1510–1521; 9/12 met the prospectively fixed teacher validation-accuracy eligibility threshold (>=0.95), exceeding the preregistered minimum of 8. All seeds were retained and no rescue seeds or bases were added.

Across eligible seeds, the worst anchored-16 basis improved the frozen recursive cluster in 9/9 seeds. The paired 100,000-resample bootstrap (RNG 20261110) gave mean D_worst -0.00982945, 95% CI [-0.01376599, -0.00645892]. The worst-basis/full-32 geometric NMSE ratio was 1.060916x, 95% CI [1.045745, 1.075494], below the locked 1.25x margin. Basis sensitivity was 1.014620x, 95% CI [1.011359, 1.017893], below 1.15x.

Validation utility difference (worst anchored basis minus full-32) averaged -1.893 percentage points with 95% CI [-2.881, -0.905] pp. Held-out test difference averaged -1.605 pp with 95% CI [-2.346, -0.864] pp. Both lower bounds remained above the preregistered -3 pp safeguard. Test evaluation occurred only after fitting and validation metrics within each eligible seed were complete.

The informative same-basis comparison also favored self-anchoring over naive 16D sketch supervision in 36/36 basis-seed pairs (mean final-NMSE difference -0.125560).

## Interpretation

C20 confirms, within this SmallViT-family central-two-block testbed, that half-dimensional hidden correction plus a complementary Canaria self-anchor can repair the recursive composition boundary robustly across fresh model seeds and prospectively fixed identity/random basis choices, with bounded functional and utility loss relative to full hidden alignment.

This does **not** establish full-model Transformer compression, arbitrary Transformer families/subspaces, LLM behavior, or a universal 16D interface. C18 remains an important negative boundary: applying the same token-wise replacement grammar across the full four-block SmallViT span collapsed task utility to chance even though the relative self-anchor ordering remained visible.
