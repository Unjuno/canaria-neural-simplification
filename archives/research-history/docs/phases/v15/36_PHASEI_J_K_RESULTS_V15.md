# Phase I/J/K results — low-bit precision, weight count, refit, and scale metadata

Exploratory follow-up; 8 paired eligible seeds 1800–1807.

## Dense quantization
- 2-bit tensor_max retention vs seed FP32 Conv3: 0.77295.
- 2-bit channel_max: 0.91139.
- 2-bit channel_calibrated: 0.96863.
- 3-bit channel_calibrated: 0.99192.
- 4-bit channel_calibrated: 0.99806.
- 12-bit: 1.00000.

Paired 2-bit gains:
- channel_max - tensor_max: +0.13844, 95% bootstrap CI [0.04692, 0.25616].
- channel_calibrated - tensor_max: +0.19568 [0.10417, 0.30830].
- channel_calibrated - channel_max: +0.05724 [0.02767, 0.08931].

## Sparse 4-bit support refit
At low K, re-solving the retained coefficients helps. Balanced support K=64 gains +0.03641 retention [0.00659, 0.06565] over inherited values. At high K (224–256), refit gives no consistent gain.

Key balanced-refit 4-bit points:
- K=96: retention 0.97345, utility 0.94841.
- K=128: retention 0.98112, utility 0.95640.
- K=192: retention 0.99163, utility 0.96675.

## K x bit nominal Pareto accounting
Nominal sparse code assumes 10 index bits per retained scalar plus 8 FP16 channel scales. This is an accounting assumption; FP16 scale storage was separately tested below.

Examples:
- balanced K=112, 3-bit: 198 B nominal, retention 0.98460.
- global K=128, 3-bit: 224 B, retention 0.98105.
- global K=192, 3-bit: 328 B, retention 0.99054.
- balanced K=224, 3-bit: 380 B, retention 0.99444.

Dense per-channel codes with 8 FP16 scales would nominally be:
- 2-bit: 162 B.
- 3-bit: 235 B.
- 4-bit: 308 B.
Thus unstructured sparsity is not automatically smaller once index cost is counted; dense 3-bit is especially competitive.

## Tau=2 repair after balanced-refit 4-bit
- K=64: utility 0.93602, PASS95 4/8.
- K=96: 0.94703, PASS95 4/8.
- K=128: 0.96113, PASS95 5/8.
- K=160: 0.95948, PASS95 4/8.
- K=192: 0.96253, PASS95 6/8.
No K in this subset achieves 8/8 stability.

## FP16 scale metadata validation
For dense 2/3/4-bit and selected sparse 3/4-bit conditions, replacing FP32 channel scales with FP16 scales changed measured retention by exactly 0.0 on all 8 seeds at the experiment's accuracy resolution. This validates using 16-bit scale metadata for the nominal byte accounting in these tested conditions.
