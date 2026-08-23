# Canaria knowledge map

This page maps research questions to the strongest currently public evidence in the repository.

## 1. What is the project actually claiming?
Read:
- `RESEARCH_SUMMARY.md`
- `CLAIMS_AND_EVIDENCE.md`
- `NEGATIVE_RESULTS.md`

Current synthesis: trained networks in the tested setting contain task-conditioned compositional redundancy. Some simplification is intrinsic; some becomes viable only after repair. Part of the lost computation is reparameterized in the remaining network, but whole-network code can still decrease under multiple operational codecs.

## 2. Is Canary the simplification law?
No evidence supports that strong interpretation.

Read:
- `history/v10/02_CONFIRMED_FINDINGS.md`
- `history/v10/21_CANARY_BLIND_DECISIVE_PROTOCOL.md`
- `../results/phaseA_v11/stage3_confirmatory_summary.json`

Key result: low-Canary regions still showed a high strong-simplification rate in the blinded confirmatory experiment. Canary's incremental AUC beyond width was small and statistically uncertain.

## 3. Does composition become simpler?
Read:
- `history/v10/01_RESEARCH_HANDOFF.md`
- `../results/history/v10/key_results.csv`
- `../results/phaseA_v11/stage3_confirmatory_summary.json`

The early 4-seed general-composition experiment found 72.5% simplification over 80 events. The later blinded confirmatory experiment found `P(G>0)=0.7107` over 3,360 events with seed-clustered uncertainty.

## 4. Is simplification intrinsic or caused by repair?
Both components exist.

Read:
- `phases/27_V11_PHASEB_SHELL_CAPACITY_RESULTS.md`
- `history/v10/25_V10_RECURSIVE_RECOMPILE_RESULTS.md`

Conv3 full-span replacement can be highly viable at tau=0, while more aggressive Conv1 collapse relies heavily on adaptive repair.

## 5. Does computation simply move downstream?
The pure-location explanation failed.

Read:
- `phases/28_PHASEC_EQUAL_CAPACITY_BOUNDARY_PROTOCOL_V12.md`
- `phases/29_PHASED_EQUAL_CAPACITY_SPATIAL_PROTOCOL_V12.md`
- `../results/v12/phaseC_equal_capacity_adapter_v12/decision.json`
- `../results/v12/phaseD_equal_capacity_spatial_v12/decision.json`
- `../results/v12/phaseE_global_boundary_adapter_v12/decision.json`

Equal-capacity local, spatial, and global adapters did not show a privileged post-boundary location. Capacity, topology, and task-output alignment matter.

## 6. Is local compression only complexity relocation?
Not under the tested whole-network codecs.

Read:
- `phases/v18/51_PHASEX_GLOBAL_ACCOUNTING_PROTOCOL_V18.md`
- `phases/v18/53_PHASEX_Y_GLOBAL_RESULTS_V18.md`
- `../results/v18/raw/phaseX_summary.json`

Whole-network reduction was ~26–29% depending on codec, with matched-control utility retained. Shell code growth accounted for only a few percent of removed-core savings.

## 7. How low can precision go?
Read:
- `phases/v15/36_PHASEI_J_K_RESULTS_V15.md`

Dense 3–4 bit quantization can preserve most of the Conv3 replacement function with calibrated/channelwise scaling. Naive 2-bit quantization is much less reliable. Reducing the number of scalar degrees of freedom is harder than reducing bit precision.

## 8. Why structured sparsity?
Read:
- `phases/v16/39_PHASEL_M_STRUCTURED_RESULTS_V16.md`

Unstructured sparse models pay explicit index overhead. Semi-structured 2:4 and structured kernel/offset patterns moved the real storage frontier substantially.

## 9. What do the 44.5 B / ~28 B numbers mean?
They are **core-only** numbers.

Read:
- `phases/v17/47_PHASEU_44B_CONFIRM_PROTOCOL_V17.md`
- `phases/v17/49_PHASEW_ENUMERATIVE_CODEC_V17.md`
- `phases/v17/50_V17_SUB100_LOWBIT_RESULTS.md`

44.5 B is an independently confirmed structured ternary replacement core after repair. 38 B and the variable ~28 B average are exact serializations of that same core, not a complete neural network.

## 10. What is the real whole-model result?
Read:
- `phases/v19/61_PHASEAB_CONFIRM_9926B_V19.md`
- `phases/v19/62_V19_HEAD_COMPRESSION_RESULTS.md`
- `../results/v19/raw_AD/confirmatory_codec_summary.json`
- `../scripts/phases/v19/run_phaseAD_exact_codec_v19.py`

The current end-to-end exact codec is **9,926 bytes**. Eight independently confirmed models had exact codec roundtrips with zero logit difference.

## 11. Why is the final whole model not using the smallest 44.5 B core?
Because local minima are not necessarily global minima.

The stable sub-10-KB whole model uses a larger 296 B Conv3-q4 core, allowing stronger compression of the task-aligned classifier while preserving seed-level stability. This motivates a network-wide complexity-allocation view.

## 12. What failed?
Read `NEGATIVE_RESULTS.md` and `history/v10/06_NEGATIVE_RESULTS_AND_PITFALLS.md`.

Major failed/weak explanations include:
- plain 8-block trainability,
- Canary as a necessary condition,
- strong incremental Canary prediction,
- pure post-location repair,
- unlimited recursive recompilation,
- 4/12-scalar representations,
- naive 2-bit quantization,
- index-heavy unstructured sparsity,
- several nominally sub-10-KB head-compression designs.

## 13. What remains unknown?
Read `ROADMAP.md`.

Highest-priority external questions:
- Fashion-MNIST / CIFAR-10 / ResNet replication,
- arbitrary graph/subgraph cuts,
- codec-independent complexity-definition sensitivity,
- off-manifold behavior,
- task-effective repair dimension,
- mechanism algebra and cross-seed canonicalization.

## 14. Where is the historical experiment inventory?
- `../results/history/v10/experiment_catalog.csv` — 156 archived experiment directories and report/file counts.
- `../results/history/v10/claim_registry.csv` — numerical and qualitative claim provenance.
- `../results/history/v10/key_results.csv` — compact historical metric index.

Historical folders/scripts were built as research artifacts rather than a stable software package; consult `../scripts/README.md` and `REPRODUCIBILITY.md` before rerunning them.
