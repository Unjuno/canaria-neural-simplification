---
name: Reproduction report
description: Report a clean reproduction attempt, including failures or deviations
title: "Reproduction: "
labels: []
assignees: []
---

## Target evidence

Which exact experiment/result are you reproducing?

- experiment/phase:
- protocol/result path or SHA256:
- repository commit/tag:

## Environment

- OS:
- Python:
- PyTorch / major libraries:
- CPU/GPU/device:
- installation command:

## Data

- dataset/source:
- split/hash if applicable:
- any unavailable historical data or substitutions:

## Command / runner

Provide the exact command(s) used.

```bash

```

## Seed / inferential unit

- seed(s):
- independently initialized models/checkpoints:
- any deviation from the original seed policy:

## Result

State the original endpoint and reproduced endpoint side-by-side.

| metric | original | reproduction |
|---|---:|---:|
| | | |

## Outcome classification

- [ ] qualitative/registered endpoint reproduced
- [ ] partial reproduction
- [ ] reproduction failed
- [ ] inconclusive due to environment/data mismatch

## Deviations

List every known change from the historical protocol. Do not silently treat a substituted setup as exact reproduction.

## Logs / artifacts

Attach or link compact logs, summaries, environment metadata, and hashes where useful.

## Interpretation

Does this result change an existing public claim in `docs/CLAIMS_AND_EVIDENCE.md`, or only the reproducibility status?

## Negative result preservation

If the reproduction failed, describe the failure directly. A failed reproduction is useful evidence and should not be hidden by retuning until it passes.
