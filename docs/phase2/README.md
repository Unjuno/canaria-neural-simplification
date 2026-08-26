# Phase 2 — precision, quantization, and repair

This page is the authoritative short index for the post-snapshot precision experiments.

The important point is not that lower precision always helps or always fails. In this model family, deployable complexity depends on **bit width, scale granularity, functional boundary, and repair procedure together**.

## Valid primary sequence

### Phase 2A — precision × composition

**PASS.** At 4-bit signed-uniform post-training quantization, the composed condition had a smaller minimum passing coded size in **8/8 fresh seeds**.

Primary result:

```text
mean log2(composed/component-wise coded size) = -0.6048
bootstrap95                                   = [-0.8754, -0.3035]
geometric coded-size ratio                   = 0.6576×
```

See [`PRECISION_COMPOSITION.md`](PRECISION_COMPOSITION.md) and `../../results/phase2/precision_composition/phase2a/`.

### Phase 2B — capacity-only rescue at 3 bit

**FAIL.** Increasing the replacement budget up to 16,384 weights did not rescue naive 3-bit per-matrix PTQ.

```text
composed rescue:       0 / 8
component-wise rescue: 0 / 8
```

This rejects the simple explanation that the 3-bit failure was only insufficient parameter count.

### Phase 2C — scale granularity

**PASS.** Row-wise/per-output-channel FP16 scales rescued 3-bit PTQ in **7/8 seeds for both topologies** at 16,384 weights.

The rescue itself is therefore not uniquely compositional. It shows that bit width alone is not enough to describe the usable representation.

### Phase 2D — short quantization-aware repair

**PASS.** With the correct activation-domain repair input, short STE/QAT-style repair can make coarse per-matrix 3-bit representations viable in this residual-MLP family.

The coded-size difference at equal 16,384 weights is only a few bytes of scale metadata; that small arithmetic difference should not be overinterpreted.

## Critical correction: Phase 2E

**Phase 2E is invalid scientific evidence.**

A forensic audit found a silent implementation bug: the replacement modules were fitted on the internal activation domain `ta[0]`, but Phase 2E repair accidentally fed raw digit inputs `Xt` to the first component and composed map. Both tensors have width 64, so the error did not trigger a shape exception.

Logically, the faulty calls were equivalent to:

```text
qforward(sr1, Xt[ix])     instead of qforward(sr1, ta[0][ix])
qforward(comp, Xt[ix])    instead of qforward(comp, ta[0][ix])
```

The original Phase 2E artifacts should be preserved and marked invalid, not deleted.

Consequences:

- Phase 2D remains valid.
- Phase 2E is **INVALIDATED_IMPLEMENTATION_BUG**.
- Phase 2F and 2G numerical experiments remain usable, but explanations framed as accounting for Phase 2E are obsolete.
- Phase 2H numerical outputs remain historical observations, but the “hard cohort” interpretation is weakened because that cohort was defined by the bugged Phase 2E result.
- The Phase 2I claim that repair RNG alone explained Phase 2E is retracted.
- The Phase 2J comparison to Phase 2E is confounded by the corrected input domain; its full-batch success remains only a numerical observation.

## Corrected follow-up

### Phase 2K

**ABORTED_PRECONDITION.** An equal-example repair benchmark was stopped after one seed once the Phase 2E premise was found to be false. The partial result is retained but excluded from inference.

### Phase 2L — controlled bug correction

**PASS.** The only intended change was the repair input `Xt -> ta[0]`; seeds, base fits, learning rate, repair RNG, checkpoints, and validation rule were otherwise held fixed.

```text
corrected composed pass:       8 / 8   (bugged Phase 2E: 0 / 8)
corrected component-wise pass: 8 / 8
composed first-pass updates:   [2, 2, 2, 4, 8, 2, 8, 2]
```

This identifies the Phase 2E failure primarily as an implementation artifact.

### Phase 2M — equal sample-presentation repair policies

**B128_EFFICIENT.** Fresh seeds 31700–31707; all methods had a 4096-sample cap.

Median sample presentations to first pass for the composed condition:

```text
random batch128            384
random batch256            768
random batch512           1536
random batch1024          3072
deterministic batch128     384
2-restart batch128         384
4-restart batch128         384
```

Fewer optimizer steps at larger batch size did not mean lower sample cost.

### Phase 2N — repair sample curve

**PASS.** Fresh `n=16`.

```text
composed:       16/16 pass by 1024 samples; all by 640
component-wise: 16/16 pass by 1024 samples; all by 896
median first-pass samples: composed 320, component-wise 384
```

The apparent composed repair-cost advantage was suggestive, not yet confirmatory.

### Phase 2O — confirmatory repair-cost advantage

**UNCERTAIN.** Fresh `n=24`.

```text
composed pass by 1024 samples:       23 / 24
component-wise pass by 1024 samples: 23 / 24
paired comparable seeds:             22
wins / ties / losses for composed:   11 / 5 / 6
median paired difference:            -64 samples
one-sided exact sign-test:            p = 0.1662
bootstrap95 mean difference:         [-157.1, +58.2]
```

Therefore the stronger statement that composition reliably reduces QAT repair sample complexity is **not confirmed**.

## Current interpretation

Supported:

1. The 4-bit compositional coded-size advantage from Phase 2A survives.
2. Naive 3-bit per-matrix PTQ is not rescued simply by adding more weights.
3. Finer scale granularity can rescue 3-bit PTQ without repair.
4. Correctly implemented short QAT-style repair can also make per-matrix 3-bit viable in this residual-MLP family.
5. In the tested fresh policy comparison, batch128 was sample-efficient.

Not supported:

1. The bugged Phase 2E 0/8 result as evidence of stochastic repair failure.
2. Repair minibatch RNG as the sole explanation for Phase 2E.
3. Deterministic full-batch repair as a demonstrated mechanism-specific fix for Phase 2E.
4. A confirmed compositional advantage in repair sample complexity.

## Evidence policy

Protocol locks and negative/invalidated results are retained. Corrections are additive: an incorrect artifact is marked and superseded rather than erased.

The consolidated local correction archive used for the 2K–2O audit has SHA256:

`1a339be12d7644de534ac77a712307c49ee0c3d9acb28c8a3532883edca3dab7`

A compact machine-readable status registry is in [`../../results/phase2/precision_composition/CORRECTION_STATUS.json`](../../results/phase2/precision_composition/CORRECTION_STATUS.json).