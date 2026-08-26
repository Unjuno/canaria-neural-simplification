# Phase 2 invalidated / superseded evidence history

This file preserves correction provenance. Invalidated evidence is retained as history but must not be cited as scientific support.

## Phase 2E — INVALIDATED_IMPLEMENTATION_BUG

**Scientific use:** `DO_NOT_USE_FOR_INFERENCE`

The repair path used raw digit input `Xt` where the replacement was defined on the internal activation domain `ta[0]`.

The semantic error was silent because both tensors had width 64. Shape compatibility therefore did not imply domain correctness.

The invalid computation was equivalent to using:

```text
qforward(sr1, Xt[ix])
qforward(comp, Xt[ix])
```

instead of:

```text
qforward(sr1, ta[0][ix])
qforward(comp, ta[0][ix])
```

### Consequences

- The Phase 2E `0/8` composed result is **not evidence** that short stochastic repair fails.
- Any causal explanation that depends on that failure must be removed or rewritten.
- Phase 2I's claim that repair RNG explained Phase 2E is retracted.
- Phase 2H observations remain post-hoc and their “hard cohort” interpretation is weakened because cohort membership depended on invalid 2E outcomes.
- Phase 2J's full-batch success remains a numerical observation only; comparison against bugged 2E is confounded by the corrected activation domain.
- Phase 2K was aborted once its premise was found invalid.

## Corrected evidence boundary

The controlled Phase 2L correction changed the intended repair input from `Xt` to `ta[0]` and reported passing composed and component-wise repair in `8/8` seeds under the controlled rerun.

Later Phase 2O did **not** confirm a reliable composed repair-sample advantage:

```text
paired comparable seeds: 22
wins / ties / losses:     11 / 5 / 6
one-sided exact sign p:   0.1662
bootstrap95 mean diff:    [-157.1, +58.2] samples
```

Therefore the public claim is viability of correctly implemented short activation-domain repair in the tested residual-MLP family, **not** lower repair-sample complexity for composition.

## Provenance

Authoritative machine-readable correction registry:

`CORRECTION_STATUS.json`

Correction archive SHA256:

`1a339be12d7644de534ac77a712307c49ee0c3d9acb28c8a3532883edca3dab7`

The full later-phase raw correction archive is not stored in this Git branch. This file records status/provenance; it does not pretend that unavailable raw artifacts are publicly reproduced here.
