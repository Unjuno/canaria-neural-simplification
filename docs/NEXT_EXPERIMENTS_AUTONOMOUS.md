# Next autonomous Canaria experiments

Date: 2026-08-24

## G18 — recontracting-aware commit policy

### Hypothesis

A controller that accounts for the **remaining task-learning horizon and expected recovery after a commit** can outperform the current static functional-NMSE threshold at equal or lower compiler cost.

### Motivation

G13–G17 show that instantaneous replacement fidelity is not sufficient to choose the best consolidation path:

- waiting until a direct 4→2 candidate becomes accurate enough was worse than staged consolidation;
- committing 4→2 very early was also worse than staged consolidation;
- staged 4→3→2 was better only when task learning occurred between the two consolidations;
- factorizing the compiler fit without intervening task learning was equivalent to direct 4→2.

The controller should therefore estimate the value of **commit now + recontract** rather than only asking whether current NMSE is below a static threshold.

### Minimal design

- Same 4-block→2-block task, data, optimizer, calibration split, and final parameter count as G11.
- Baseline: stable G11 v2 static-NMSE policy.
- New policy: candidate score uses only non-test information:
  - current held-out functional NMSE;
  - parameter reduction;
  - compiler effort already spent;
  - remaining task epochs;
  - online estimate of recovery slope from prior commit events in the same run.
- The exact score function must be frozen before fresh seeds.
- Test data remains completely outside commit decisions.

### Primary endpoint

Final held-out test PPL at fixed final parameter count.

### Secondary endpoints

- total compiler parameter-update proxy;
- final commit epoch;
- number of rejected candidate checks;
- recovery slope after each commit;
- fraction of seeds that reach the target final architecture.

### Decision sketch

PASS should require the new policy to improve final PPL versus the static G11 controller without increasing compiler cost beyond a preregistered tolerance. If the new policy reduces compiler effort materially at equivalent PPL, that should be treated as a separate Pareto success rather than collapsed into the same endpoint.

## G19 — generalize the staged-path effect

### Hypothesis

The G15/G17 result is not specific to the 4→3→2 path. A different source depth and intermediate path should show the same qualitative effect: **task learning between consolidation events**, rather than fit factorization alone, produces the staged advantage.

### Minimal design

Example:

- source: 5-block core;
- final: 2-block core;
- staged: 5→4→3→2 with task learning between commits;
- direct: 5→2;
- factorized-no-learning control: 5→4→3→2 compiler sequence performed without task-learning intervals, with matched final-commit epoch and compiler cost.

Use a fresh model-depth configuration or model family and fresh confirmatory seeds.

### Required interpretation rule

Do not call the staged mechanism architecture-independent unless:

1. staged with learning beats direct on fresh seeds; and
2. factorized-without-learning fails to reproduce the staged advantage or is equivalent to direct under a preregistered band.

## G20 — direct measurement of recontracting

If G18/G19 remain positive, directly measure what changes after a commit rather than inferring recontracting only from utility recovery.

Candidate measurements:

- representational similarity before/after commit and after recovery;
- Jacobian / sensitivity redistribution across the replacement boundary;
- effective rank of boundary activations;
- layer/block contribution redistribution;
- whether the next compiler fit becomes easier after an intermediate task-learning interval;
- task-conditioned functional description length of the span before and after recovery.

The highest-value version of G20 should explicitly distinguish:

- ordinary architecture regularization;
- optimizer-basin migration;
- representation redistribution;
- genuine simplification of the task-conditioned span function.

## General experimental rules

- Fresh seeds for confirmatory cohorts.
- Protocol and decision rule frozen before outcome inspection.
- Test data excluded from controller decisions.
- Model seed is the inference unit unless otherwise preregistered.
- Compiler compute, task compute, parameter count, and serialized size reported separately.
- Negative results and failed candidate policies retained.
- A result on the current character-LM testbed is not automatically generalized to pretrained Transformers.
