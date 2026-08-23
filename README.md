# Canaria Neural Simplification

**Canaria** is an experimental research repository for studying task-conditioned computational simplification, redistribution, compilation, and low-bit compression in trained neural networks.

> **Project status (2026-08-24): experiments paused; repository curation and reproducibility phase.**

The repository contains the experiment code, preregistered/locked protocols, confirmatory summaries, negative results, historical experiment reports, and low-bit codecs developed during the current research program.

## What has been established so far

The strongest current findings are:

- **Composition simplification is not confined to high-Canary regions.** In an 8-seed blinded confirmatory test with 3,360 composition events, low-Canary strong-simplification rate was **0.845** (95% seed-cluster CI **0.7225–0.9500**). The hypothesis that Canary is a necessary local condition failed.
- **Subadditivity is common in the tested setting.** The confirmatory composition-subadditivity rate was **0.7107** (95% CI **0.6128–0.8137**).
- **The tested Canary signal adds little predictive information beyond span width.** LOSO AUC improvement was only **+0.00567**, with a 95% CI crossing zero. Canary should currently be treated as a weak/uncertain sensor, not a causal driver.
- **Pure downstream-location explanations failed.** Equal-capacity pre/post interventions (local, spatial, and global adapters) repeatedly found no special advantage for the post boundary after capacity/function-class control.
- **Global simplification survives whole-network accounting.** Relative to matched continued-training controls, compiled models showed about **26.1% FP32** and **28.8% q8+zlib** whole-network reduction at utility **0.988**. Shell code growth offset only a few percent of removed-core savings under these codecs.
- **Extreme core compression is possible, but core-only numbers are not whole-model sizes.** A 44.5-byte core was independently confirmed after repair; exact ternary/enumerative serialization reduced the same core representation further without changing predictions.
- **A real whole-network codec below 10 KB was independently confirmed.** The current best exact serialized model is **9,926 bytes**, using a 4-bit Conv3 core and a 2:4 structured sparse classifier head. Pack/unpack was bit-exact at the model-output level, with confirmatory combined fidelity **0.9636** (95% CI **0.9516–0.9740**) and matched-control utility **0.9835** (95% CI **0.9639–1.0059**).

These claims are **architecture/task conditioned**. They are not yet a universal law of neural networks.

## Repository map

- [`docs/RESEARCH_SUMMARY.md`](docs/RESEARCH_SUMMARY.md) — current integrated research narrative.
- [`docs/CLAIMS_AND_EVIDENCE.md`](docs/CLAIMS_AND_EVIDENCE.md) — claim-by-claim evidence strength and limitations.
- [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) — failed hypotheses and why they matter.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — environment, seed policy, blindness/hash-lock rules, and reproduction guidance.
- [`docs/phases/`](docs/phases/) — phase-specific protocols and result reports.
- [`scripts/history/v10/`](scripts/history/v10/) — the historical experiment suite (102 Python scripts).
- [`scripts/phases/`](scripts/phases/) — later confirmatory, causal-intervention, quantization, structured-sparsity, global-accounting, and codec scripts.
- [`results/`](results/) — locked tables, confirmatory summaries, per-seed result CSV/JSON where available, and historical result indices.
- [`docs/experiment-reports/v10/`](docs/experiment-reports/v10/) — 100+ historical experiment reports from the first research program.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/audit_repo.py
```

Most historical scripts were research scripts rather than a stable library API. Start with the phase protocol associated with the script you want to reproduce; do not assume every script has identical CLI conventions.

## Evidence classes used here

- **Confirmatory** — condition/protocol locked before outcome inspection; independent seeds and explicit decision rule.
- **Independent holdout** — condition selected previously, then re-tested on new seeds without re-selection.
- **Pilot / exploratory** — hypothesis-generating or implementation-validating experiment.
- **Negative result** — a failed hypothesis retained as first-class evidence rather than discarded.

## Key limitations

1. Most decisive experiments currently use a digits-like task and an 8-block residual CNN.
2. Operational complexity depends on the candidate grammar/codec; it is not a proof of Kolmogorov complexity or codec-independent minimum description length.
3. The 44.5-byte / ~28-byte numbers refer to a **compiled core**, not a complete network.
4. The 9,926-byte result is a **whole-network** serialized model and is the correct number to use for the current end-to-end compression claim.
5. Historical package versions were not fully pinned in the earliest experiments; see reproducibility notes.

## License

Code and original documentation in this repository are released under the **Apache License 2.0**. Third-party datasets, libraries, or copied third-party materials remain under their own licenses.

## Citation

See [`CITATION.cff`](CITATION.cff). Until a paper is published, cite this repository/version and the relevant experiment protocol/result file.
