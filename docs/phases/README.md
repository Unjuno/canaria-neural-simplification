# Phase index

This directory is the chronological record of the later research program. Protocols should be read before result files for the same phase. A result selected after looking at an earlier cohort is exploratory until it is re-tested on a fresh cohort.

| Phase/version | Main question | Evidence role | Key files |
|---|---|---|---|
| v11 Phase A | Blind simplification map; Canary observer vs local condition | Confirmatory | historical `21_CANARY_BLIND_DECISIVE_PROTOCOL.md`, locked results under `results/phaseA_v11/` |
| v11 Phase B | Shell locus/capacity | Confirmatory + follow-up | `27_V11_PHASEB_SHELL_CAPACITY_RESULTS.md` |
| v12 C/D/E | Pure location vs capacity/function class | New-seed causal controls | `28_...`, `29_...`, `30_...`, results under `results/v12/` |
| v13 G | Float-count / top-K weight budget | Exploratory measurement | `v13/32_PHASEG_FLOAT_BUDGET_PROTOCOL_V13.md` |
| v14 H | Precision × weight-count threshold | New-seed sweep | `v14/33_...`, `v14/34_...` |
| v15 I/J/K | Quantizer choice, sparse refit, storage frontier, scale precision | New-seed + paired analysis | `v15/35_...`, `v15/36_...` |
| v16 L/M | Structured sparsity and independent holdout | Exploration + independent holdout | `v16/37_...`, `v16/38_...`, `v16/39_...` |
| v17 N–W | Sub-100-byte core, scale/pattern sharing, independent confirmation, exact codecs | Mixed; key claims independently confirmed | `v17/40_...` through `v17/50_...` |
| v18 X/Y | Whole-network accounting and low-bit whole-model boundary | New-seed matched accounting | `v18/51_...` through `v18/53_...` |
| v19 Z–AD | Head compression and exact 9,926-byte whole-network codec | Exploration followed by independent confirmation | `v19/54_...` through `v19/62_...` |

## Evidence discipline

- `protocol` means the condition was written before the associated outcome was inspected.
- `confirm` / `holdout` means a previously selected condition was evaluated on fresh seeds without reselection.
- `exploratory` means the condition was motivated by observed results and must not be presented as preregistered evidence.
- Exact serialization claims require pack/unpack identity checks, not only nominal bit-count calculations.

For the current cross-phase interpretation, use `../RESEARCH_SUMMARY.md` and `../CLAIMS_AND_EVIDENCE.md` rather than reading one phase in isolation.