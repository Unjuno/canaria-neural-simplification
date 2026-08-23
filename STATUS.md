# Project status

**2026-08-24: active experiment execution is paused.** The repository is in a curation/reproducibility phase.

## Public curation state

The public `main` branch now contains:

- the integrated current research narrative;
- claim/evidence and negative-result registers;
- the historical v10 handoff, research timeline, terms, evidence matrix, and blind confirmatory protocol;
- a 156-experiment historical catalog and verified historical key-results/claim registry;
- locked Phase-A confirmatory summaries;
- equal-capacity causal-control decisions;
- low-bit / structured-sparsity / sub-100-byte core result reports;
- whole-network global-accounting results;
- the independently confirmed 9,926-byte whole-network protocol, summary, and original exact bit-packing implementation;
- Apache-2.0 license, citation metadata, contribution rules, CI audit, and reproducibility guidance.

The curated public tree is the canonical human-readable research record. Large duplicated historical ZIPs, generated plots/caches, and training checkpoints are intentionally not copied into Git history. The historical experiment catalog and claim registry preserve the inventory/provenance of those archived runs.

## Repository policy while experiments are paused

- Do not overwrite locked protocols or historical result files.
- Keep failed hypotheses and non-replications visible.
- Separate confirmatory, independent-holdout, pilot, and exploratory evidence.
- Report uncertainty at the seed/model cluster level where repeated spans/events share a network.
- Distinguish core bytes, nominal code length, entropy estimates, and real whole-model serialized bytes.
- Port old absolute-path scripts additively rather than rewriting the historical evidence chain.
