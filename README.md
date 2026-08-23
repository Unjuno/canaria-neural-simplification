# Canaria Neural Simplification

**Canaria** is an experimental research repository for studying task-conditioned computational simplification, redistribution, compilation, and low-bit compression in trained neural networks.

> **Project status (2026-08-24): experiments paused; repository curation and reproducibility phase.**

The goal of the current phase is not to generate new positive results. It is to make the accumulated evidence—confirmatory, exploratory, and negative—auditable and reusable by other researchers.

## Current strongest findings

- **Simplification is not confined to high-Canary regions.** In the blinded 8-seed confirmatory Phase A (3,360 composition events), low-Canary strong-simplification rate was **0.845** (95% seed-cluster CI **0.7225–0.9500**). A Canary-local necessary-condition hypothesis failed.
- **Composition subadditivity is common in the tested setting.** Confirmatory `P(G>0)` was **0.7107** (95% CI **0.6128–0.8137**).
- **The tested Canary adds little predictive value beyond span width.** LOSO AUC improvement was **+0.00567**, with a 95% CI crossing zero. Treat the current Canary as a weak/uncertain sensor, not a causal law.
- **Pure post-boundary location explanations failed.** Three equal-capacity pre/post intervention families found no special post-location advantage after capacity/function-class control.
- **Whole-network simplification survives accounting.** Versus matched continued-training controls, compiled models were about **26.1% smaller in fixed FP32 code** and **28.8% smaller under q8+zlib**, while mean utility was **0.988**. Measured shell-code growth offset only a few percent of removed-core savings.
- **Extreme core compression is possible, but core bytes are not model bytes.** A **44.5-byte** structured ternary core was independently confirmed after repair; exact serialization of the same core could be smaller without changing predictions.
- **A real whole-network codec below 10 KB was independently confirmed.** The current end-to-end result is **9,926 bytes**, with exact pack/unpack (`max logit diff = 0`), combined fidelity **0.9636** (95% CI **0.9516–0.9740**) and matched-control utility **0.9835** (95% CI **0.9639–1.0059**).

These findings are **task/architecture conditioned**. They are not a universal theorem about neural networks.

## Start here

1. [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md) — integrated current narrative.
2. [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md) — claim-by-claim status and limitations.
3. [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) — failed explanations retained as evidence.
4. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — statistical unit, blind-lock procedure, seed policy, storage terminology.
5. [`docs/ROADMAP.md`](docs/ROADMAP.md) — open questions for when experiments resume.

## Research history

The historical v10 handoff captures how the research moved from Canary-guided local compression to a broader theory of adaptive compositional simplification:

- [`docs/history/v10/00_START_HERE.md`](docs/history/v10/00_START_HERE.md)
- [`docs/history/v10/01_RESEARCH_HANDOFF.md`](docs/history/v10/01_RESEARCH_HANDOFF.md)
- [`docs/history/v10/02_CONFIRMED_FINDINGS.md`](docs/history/v10/02_CONFIRMED_FINDINGS.md)
- [`docs/history/v10/05_TERMS_AND_METRICS.md`](docs/history/v10/05_TERMS_AND_METRICS.md)
- [`docs/history/v10/06_NEGATIVE_RESULTS_AND_PITFALLS.md`](docs/history/v10/06_NEGATIVE_RESULTS_AND_PITFALLS.md)
- [`docs/history/v10/09_RESEARCH_TIMELINE.md`](docs/history/v10/09_RESEARCH_TIMELINE.md)
- [`docs/history/v10/10_EVIDENCE_MATRIX.md`](docs/history/v10/10_EVIDENCE_MATRIX.md)
- [`docs/history/v10/13_REPRODUCTION_GUIDE.md`](docs/history/v10/13_REPRODUCTION_GUIDE.md)
- [`docs/history/v10/21_CANARY_BLIND_DECISIVE_PROTOCOL.md`](docs/history/v10/21_CANARY_BLIND_DECISIVE_PROTOCOL.md)
- [`docs/history/v10/25_V10_RECURSIVE_RECOMPILE_RESULTS.md`](docs/history/v10/25_V10_RECURSIVE_RECOMPILE_RESULTS.md)

Historical numeric claims are indexed in:

- [`results/history/v10/key_results.csv`](results/history/v10/key_results.csv)
- [`results/history/v10/claim_registry.csv`](results/history/v10/claim_registry.csv)

## Decisive later phases

### Blind simplification / Canary
- [`results/phaseA_v11/STAGE1_LOCK.json`](results/phaseA_v11/STAGE1_LOCK.json)
- [`results/phaseA_v11/stage3_confirmatory_summary.json`](results/phaseA_v11/stage3_confirmatory_summary.json)

### Equal-capacity causal controls
- [`results/v12/phaseC_equal_capacity_adapter_v12/decision.json`](results/v12/phaseC_equal_capacity_adapter_v12/decision.json)
- [`results/v12/phaseD_equal_capacity_spatial_v12/decision.json`](results/v12/phaseD_equal_capacity_spatial_v12/decision.json)
- [`results/v12/phaseE_global_boundary_adapter_v12/decision.json`](results/v12/phaseE_global_boundary_adapter_v12/decision.json)

### Extreme core compression
- [`docs/phases/v17/47_PHASEU_44B_CONFIRM_PROTOCOL_V17.md`](docs/phases/v17/47_PHASEU_44B_CONFIRM_PROTOCOL_V17.md)
- [`docs/phases/v17/49_PHASEW_ENUMERATIVE_CODEC_V17.md`](docs/phases/v17/49_PHASEW_ENUMERATIVE_CODEC_V17.md)
- [`docs/phases/v17/50_V17_SUB100_LOWBIT_RESULTS.md`](docs/phases/v17/50_V17_SUB100_LOWBIT_RESULTS.md)

### Global accounting
- [`docs/phases/v18/51_PHASEX_GLOBAL_ACCOUNTING_PROTOCOL_V18.md`](docs/phases/v18/51_PHASEX_GLOBAL_ACCOUNTING_PROTOCOL_V18.md)
- [`docs/phases/v18/53_PHASEX_Y_GLOBAL_RESULTS_V18.md`](docs/phases/v18/53_PHASEX_Y_GLOBAL_RESULTS_V18.md)
- [`results/v18/raw/phaseX_summary.json`](results/v18/raw/phaseX_summary.json)
- [`results/v18/raw_phaseY/phaseY_summary.json`](results/v18/raw_phaseY/phaseY_summary.json)

### Exact whole-network codec
- [`docs/phases/v19/61_PHASEAB_CONFIRM_9926B_V19.md`](docs/phases/v19/61_PHASEAB_CONFIRM_9926B_V19.md)
- [`docs/phases/v19/62_V19_HEAD_COMPRESSION_RESULTS.md`](docs/phases/v19/62_V19_HEAD_COMPRESSION_RESULTS.md)
- [`results/v19/raw_AD/confirmatory_codec_summary.json`](results/v19/raw_AD/confirmatory_codec_summary.json)
- [`scripts/phases/v19/run_phaseAD_exact_codec_v19.py`](scripts/phases/v19/run_phaseAD_exact_codec_v19.py) — original exact bit-packing implementation; see [`scripts/README.md`](scripts/README.md) for historical path assumptions.

## Evidence classes

- **Confirmatory** — condition/protocol locked before outcome inspection; independent seeds and explicit decision rule.
- **Independent holdout** — condition selected earlier, then retested on new seeds without reselection.
- **Pilot / exploratory** — hypothesis-generating or implementation-validating.
- **Negative result** — failed hypothesis kept as first-class evidence.

## Quick environment check

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/audit_repo.py
```

Historical experiments did not preserve a single exact dependency lockfile. Read [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before attempting bitwise reproduction.

## Important scope limits

1. Most decisive experiments use a digits-like task and an 8-block residual CNN.
2. Operational complexity depends on the candidate grammar/codec; this is not a proof of Kolmogorov complexity or codec-independent minimum description length.
3. **44.5 B / ~28 B are compiled-core results, not whole-network sizes.**
4. **9,926 B is a real whole-network serialized size** and is the correct number for the current end-to-end compression claim.
5. Cross-dataset, ResNet, and arbitrary-subgraph external validity remain open.

## License

Original code and documentation are released under the **Apache License 2.0**. Third-party datasets/libraries remain under their own licenses.

## Citation

See [`CITATION.cff`](CITATION.cff). Until a paper is published, cite the repository commit/tag and the exact protocol/result files used.
