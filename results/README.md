# Results index

`results/` contains the **current machine-readable evidence surface**. The older v10–v25 research sequence is preserved under `../archives/research-history/results/` and is no longer first-class current navigation.

## Reviewed evidence

- `core_discovery_digits/`
  - residual-MLP direct component-wise/composed replication;
  - exact learned replacement-parameter matching;
  - locked protocol and confirmatory summary.
- `replication/vit_compositional/`
  - SmallViT direct replication;
  - read the current documentation for its test-recording caveat.
- `training_time/`
  - reviewed training-time protocol/summary material.
- `phase2/precision_composition/`
  - reviewed precision/quantization evidence;
  - **read `CORRECTION_STATUS.json` and `INVALIDATED_HISTORY.md` before using later Phase 2 material**.
- `reproduction/`
  - portability/reproduction and bounded systems reports;
  - reproduction does not add new confirmatory scientific seeds.

## Status rule

A file in `results/` is not automatically a headline claim. Interpretation is controlled by:

- `../docs/CLAIMS_AND_EVIDENCE.md`;
- `../docs/ANNOUNCEMENT_READINESS.md`;
- relevant protocol locks, correction records, and review documents.

Likewise, a historical result under `../archives/research-history/results/` may be valid evidence for a narrow historical question while not belonging in the current announcement claim set.

## Correction / immutability policy

Once an artifact has been used for inference, do not silently replace it. If a defect is found:

1. preserve the original artifact where feasible;
2. add an explicit correction/invalidation record;
3. remove or narrow dependent claims;
4. rerun under a new corrected protocol/version if scientific re-evaluation is required.

Phase 2E is the canonical example: the invalid artifact remains provenance, while current interpretation marks it `INVALIDATED_IMPLEMENTATION_BUG` / `DO_NOT_USE_FOR_INFERENCE`.

## Historical sequence

The older version-number sequence, including `phaseA_v11`, `phaseB_v11`, and `v12` through `v25`, moved without scientific reinterpretation to:

`../archives/research-history/results/`

See `../archives/README.md` for the migration map. Archived paths may contain obsolete cross-links and environment assumptions; use them for provenance, not current navigation.
