# Reproduction and systems-PoC evidence

This directory contains evidence that is **not a new scientific confirmatory cohort**, but validates portability or a bounded engineering hypothesis.

## 1. G7 exact public reproduction

Files:

- `g7_seed4300_report.json`
- runner: `../../scripts/reproduce/g7_confirmatory/run_seed.py`
- instructions: `../../scripts/reproduce/g7_confirmatory/README.md`
- pinned environment: `../../scripts/reproduce/g7_confirmatory/requirements.txt`
- manual workflow: `../../.github/workflows/reproduce-g7.yml`

Result:

- fresh confirmatory seed 4300 was rerun without private `/mnt/data` imports;
- in the recorded environment, the complete JSON exactly matched the archived output;
- SHA256: `68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`.

Evidence class: **reproduction / portability**.

It does not add a new independent scientific seed.

## 2. Runtime/materialization PoC

Files:

- `runtime_poc_seed4300_report.json`
- benchmark: `../../scripts/reproduce/g7_confirmatory/runtime_poc.py`
- interpretation: `../../docs/RUNTIME_POC.md`
- manual workflow: `../../.github/workflows/runtime-poc.yml`

Recorded small CPU result:

- serialized artifact + manifest: `110,093 → 54,646 B` (`−50.36%`);
- parameters: `23,138 → 11,042` (`−52.28%`);
- batch-128 CPU inference: `47.05 → 23.11 ms mean`, five fresh-process probes;
- process RSS delta: `4.72 → 4.56 MB`, so meaningful host-RAM reduction was **not demonstrated**.

Evidence class: **systems PoC / PASS_WITH_BOUNDARY**.

The result is limited to the measured small CPU setup and must not be promoted to a universal GPU/LLM/runtime or RAM claim.

## Evidence discipline

Keep these classes distinct:

- **confirmatory science** — fresh locked scientific evaluation;
- **reproduction** — reruns an already-observed condition to test portability/reproducibility;
- **systems PoC** — tests an engineering path in one bounded environment;
- **negative/boundary** — records a failed or unsupported interpretation.

The repository audit preserves the exact G7 reproduction hash and the runtime-PoC host-RAM/generalization boundaries.
