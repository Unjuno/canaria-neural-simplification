# Phase 2 — precision, quantization, and repair

This page is the authoritative short index for the post-snapshot precision experiments after the independent pre-publication re-review.

The public conclusion is narrow: in this residual-MLP family, deployable replacement complexity depends jointly on **bit width, scale granularity, functional boundary, and repair procedure**. Lower precision is not automatically better, and composition does not receive a blanket quantization advantage.

## Public evidence boundary

Raw protocol/result files for Phase 2A–C and portable A–C runners are checked into this branch. Later correction work (2D–2O) is summarized by the machine-readable correction registry and the identified correction archive, but **not all later raw per-seed artifacts are present in Git**. Therefore later-phase statements below are correction/provenance claims, not a claim that every later phase is publicly rerunnable from this branch.

Correction archive SHA256:

`1a339be12d7644de534ac77a712307c49ee0c3d9acb28c8a3532883edca3dab7`

See:

- `../../results/phase2/precision_composition/CORRECTION_STATUS.json`
- `../../results/phase2/precision_composition/INVALIDATED_HISTORY.md`

## Valid primary sequence

### Phase 2A — precision × composition

**VALID_PASS.** At 4-bit signed-uniform post-training quantization, the composed condition had a smaller minimum passing coded size in **8/8 fresh seeds** under the locked residual-MLP protocol.

```text
mean log2(composed/component-wise coded size) = -0.6048
bootstrap95                                   = [-0.8754, -0.3035]
geometric coded-size ratio                   = 0.6576×
```

This is an operational coded-size result under the declared quantizer and FP16 scale-metadata accounting, not an intrinsic bit-complexity law.

See `PRECISION_COMPOSITION.md` and `../../results/phase2/precision_composition/phase2a/`.

### Phase 2B — capacity-only rescue at 3 bit

**VALID_FAIL.** Increasing the replacement budget up to 16,384 weights did not rescue naive 3-bit per-matrix PTQ.

```text
composed rescue:       0 / 8
component-wise rescue: 0 / 8
```

This rejects the tested capacity-only rescue hypothesis; it does not prove that every 3-bit quantizer must fail.

### Phase 2C — scale granularity

**VALID_PASS.** Row-wise/per-output-channel FP16 scales rescued 3-bit PTQ in **7/8 seeds for both topologies** at 16,384 weights.

The rescue is not uniquely compositional. It shows that nominal bit width alone is insufficient to describe usable representation cost.

## Critical invalidation — Phase 2E

**Phase 2E is INVALIDATED_IMPLEMENTATION_BUG and must not be used for scientific inference.**

The replacement modules were fitted on internal activation `ta[0]`, but the repair path accidentally fed raw digit input `Xt` to the first component and composed map. Both tensors had width 64, so the semantic-domain error passed shape checks.

The faulty calls were equivalent to:

```text
qforward(sr1, Xt[ix])
qforward(comp, Xt[ix])
```

instead of:

```text
qforward(sr1, ta[0][ix])
qforward(comp, ta[0][ix])
```

Consequences:

- the Phase 2E `0/8` composed result is **not evidence** of stochastic repair failure;
- Phase 2I's claim that repair RNG explains Phase 2E is **retracted**;
- Phase 2H's bug-defined “hard cohort” interpretation is weakened;
- Phase 2J's full-batch success remains a numerical observation, but comparison to bugged 2E is confounded;
- Phase 2K was aborted once its premise was found invalid.

The invalidated evidence is preserved as correction history rather than erased.

## Corrected later boundary

### Phase 2D / 2L — correctly implemented short repair

Correct activation-domain STE/QAT-style repair can make coarse per-matrix 3-bit representations viable in this tested residual-MLP family.

The controlled Phase 2L correction changed the intended repair input `Xt -> ta[0]` while keeping the other controlled settings fixed and reported:

```text
corrected composed pass:       8 / 8
corrected component-wise pass: 8 / 8
composed first-pass updates:   [2, 2, 2, 4, 8, 2, 8, 2]
```

This identifies the Phase 2E failure as an implementation artifact. It does not establish a universal QAT repair guarantee.

### Phase 2M — equal sample-presentation policies

Fresh seeds `31700–31707` under a 4096-sample cap found batch-128 policies sample-efficient in the tested comparison. This is a policy result for this setup, not a universal optimal batch-size claim.

### Phase 2N — exploratory repair-sample curve

Fresh `n=16` suggested lower median first-pass samples for composed repair (`320` versus `384`), but this was not yet confirmatory.

### Phase 2O — confirmatory repair-cost advantage

**VALID_UNCERTAIN.** Fresh `n=24` did not establish a reliable composed repair-sample advantage.

```text
composed pass by 1024 samples:       23 / 24
component-wise pass by 1024 samples: 23 / 24
paired comparable seeds:             22
wins / ties / losses for composed:   11 / 5 / 6
median paired difference:            -64 samples
one-sided exact sign-test:            p = 0.1662
bootstrap95 mean difference:         [-157.1, +58.2]
```

Therefore **do not claim that composition reliably lowers QAT repair sample complexity**.

## Current supported interpretation

Supported:

1. Phase 2A: the 4-bit composed coded-size advantage survives under the locked residual-MLP experiment.
2. Phase 2B: increasing weight count alone did not rescue the tested naive 3-bit per-matrix PTQ.
3. Phase 2C: finer scale granularity can rescue 3-bit PTQ for both topologies.
4. Correctly implemented short activation-domain QAT-style repair can make coarse per-matrix 3-bit viable in the tested family.

Not supported:

1. the bugged Phase 2E `0/8` result as scientific negative evidence;
2. repair minibatch RNG as the sole explanation for Phase 2E;
3. a mechanism-specific claim that deterministic full-batch repair uniquely fixes Phase 2E;
4. a confirmed compositional advantage in repair sample complexity.

## Evidence policy

Corrections are additive. Invalidated or failed results are marked and preserved; they are not silently deleted or rewritten as successes.
