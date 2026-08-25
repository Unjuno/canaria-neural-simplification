---
name: Research question / replication proposal
description: Propose a bounded experiment that distinguishes an existing claim or competing explanation
title: "Research: "
labels: []
assignees: []
---

## Existing claim or open question

Point to the exact current item in one of:

- `docs/CLAIMS_AND_EVIDENCE.md`
- `docs/OPEN_QUESTIONS.md`
- `docs/ROADMAP.md`

Do not start from an unbounded request to "try more architectures."

## Hypothesis

Write a falsifiable statement.

**H:**

## Competing explanation

What alternative interpretation will this experiment distinguish?

## Evidence class

Choose **before** outcome inspection:

- [ ] exploratory / pilot
- [ ] confirmatory
- [ ] independent holdout
- [ ] reproduction
- [ ] deployment proof-of-concept

## Minimal protocol

**T:**

- architecture/task:
- intervention:
- controls:
- calibration/validation/test separation:
- seed/checkpoint policy:
- optimization/repair/compiler budget:

## Decision rule

**D:**

State the primary endpoint, threshold/equivalence margin, uncertainty method, and PASS/FAIL/UNCERTAIN rule before fresh outcomes are inspected.

## Failure modes / confounds

**C:**

Examples: architecture mismatch, capacity mismatch, unequal optimization budget, hidden complexity relocation, test-set leakage, post-hoc retuning, correlated within-seed observations.

## Uncertainty / scope

**U:**

What will remain unproven even if this experiment passes?

## Cost justification

Why is this experiment worth running in the public-snapshot phase?

It should close at least one of:

- [ ] public-claim evidence gap
- [ ] reproducibility gap
- [ ] deployment evidence gap

## Preservation plan

List the protocol lock, code hash, run metadata, summary artifact, and negative-result documentation that will be retained.
