# Integrated research summary

## Research question

The program began with a local diagnostic question—whether a Canary-like boundary signal identifies places where a trained neural network can be replaced by simpler functions—and progressively shifted to the more general question:

> **How does task-effective computation in a trained neural network simplify, redistribute, and reparameterize under composition, replacement, repair, and compression?**

The current evidence supports a mixed picture: there is genuine net simplification under several operational codecs, adaptive compensation can increase simplification, and some computation is redistributed into the remaining network; however, redistribution is not large enough under the measured codecs to explain away the observed whole-network reductions.

## 1. Canary and blind simplification

The decisive Canary experiment used an 8-block residual CNN, all 36 contiguous spans, repair budgets τ ∈ {0,1,2,4,8}, eight confirmatory seeds, and a Stage-1/Stage-2 blindness barrier. Stage 1 selected minimal viable candidates without any Canary measurement and was SHA256 locked before Canary was computed.

Across 3,360 composition events:

- low-Canary strong-simplification rate: **0.845** (95% CI 0.7225–0.9500),
- composition-subadditivity rate: **0.7107** (95% CI 0.6128–0.8137),
- strong-simplification rate: **0.6491** (95% CI 0.5315–0.7786),
- width-only LOSO AUC: **0.7357**,
- width+Canary AUC: **0.7414**,
- ΔAUC: **+0.00567**, CI crossing zero.

Interpretation: simplification is not a Canary-local phenomenon in this setting. The tested Canary is not a necessary condition and has only weak/uncertain incremental predictive value after width.

## 2. Intrinsic vs adaptive simplification

Simplification existed at τ=0, but became much more frequent after repair. This supports a decomposition into:

1. **intrinsic replaceability/subadditivity**, present without shell adaptation, and
2. **adaptive collapse**, where remaining trainable capacity learns to compensate for a more aggressive replacement.

The effect is strongly conditioned on replacement operator complexity. Full-span Conv3 replacement can often preserve function without repair, whereas a more aggressive Conv1 collapse requires substantially more adaptation.

## 3. Capacity, location, and topology

Early post-shell repair experiments showed strong downstream recovery, but the post region also contained far more trainable parameters. Subsequent causal controls matched capacity and function class across pre/post locations.

Three equal-capacity intervention families all rejected a pure post-location advantage:

- local 1×1 boundary adapters: post−pre = **−0.0204**, 95% CI [−0.0266, −0.0138],
- ~20.7k-parameter spatial adapters: post−pre = **−0.0299**, CI [−0.0498, −0.0144],
- ~25.1k-parameter global MLP adapters: post−pre = **−0.00908**, CI [−0.0172, −0.0010].

Thus, earlier downstream recovery cannot be interpreted as a pure spatial-location effect. Parameter topology, task-output alignment, and effective repair subspace matter.

## 4. Recursive compilation and a complexity floor

The first large core collapse can work well, but repeatedly recompiling the repaired shell into a tiny linear/small-MLP representation generally failed to preserve matched-control utility. This argues against an unlimited recursive-collapse law. A better working hypothesis is a **task-conditioned complexity floor**: large redundant regions can collapse, but some nontrivial task-effective computation remains distributed in the residual network.

## 5. Global complexity accounting

A major alternative hypothesis was that local compression merely relocates an equal amount of complexity into the shell. Phase X compared compiled+repaired models against matched continued-training controls using several operational codecs.

Mean whole-network reductions:

- FP32 fixed code: **26.14%**,
- q8 ideal entropy: **27.65%**,
- q8+zlib: **28.79%** (95% CI 28.27–29.37%),
- utility: **0.9884** (95% CI 0.9716–1.0050).

Shell q8 code growth offset only roughly a few percent of the code removed from the core. Therefore, under the tested codecs, the evidence favors **net global simplification plus limited redistribution**, not pure complexity relocation.

This is not a proof of codec-independent MDL reduction.

## 6. Precision and weight-count experiments

Bit precision and number of free scalar weights behave very differently.

- Dense 12-bit and 8-bit versions were essentially lossless relative to FP32 Conv3 in the tested setting.
- 4-bit remained strong; 3-bit became viable with channelwise/calibrated quantization.
- 2-bit was highly quantizer-dependent.
- Reducing the number of scalar degrees of freedom was much more damaging than reducing precision. Four or twelve scalar weights were nowhere near sufficient; practical no-repair thresholds were initially in the hundreds of retained weights.

## 7. Structured sparsity and sub-100-byte cores

Unstructured sparsity paid too much index overhead. Structured/semi-structured schemes improved the storage/utility frontier.

Important milestones:

- 2:4 × 3-bit structured core: **181 B**, strong no-repair retention across independent holdout,
- kernel-block 24 × 3-bit: **108 B**, weaker no-repair but ~0.97 matched-control utility after short repair in two cohorts,
- 1:4 × 2-bit + shared scales: **76 B**, independently confirmed with longer repair,
- shared 1-of-4 pattern: **44.5 B**, independently confirmed after repair.

The 44.5-byte model is a core representation, not a whole network.

Because the 2-bit code was actually ternary, exact serialization could exploit unused states and zero bias. The same model could be serialized exactly in smaller byte counts without changing predictions; an enumerative zero-aware codec produced an average core payload around the high-20-byte range in the tested seeds.

## 8. Whole-network low-bit compression

After core compression, the remaining classifier/head dominates storage. Whole-network quantization showed:

- 3-bit whole-network: ~**9.95 KB**, but self-fidelity failed,
- 4-bit whole-network: ~**13.25 KB**, self-fidelity and matched-control utility passed,
- 6/8-bit: also passed at larger sizes.

Thus 4-bit was the first stable whole-network low-bit operating point before head-specific compression.

## 9. Whole-network model below 10 KB

Head compression experiments tested low-rank and structured approaches. Several nominally sub-10-KB designs preserved the compressed head itself but failed whole-network utility because seed-level core/model stability was insufficient.

The independently confirmed solution used:

- Conv3 core quantized to 4-bit (larger than the extreme 44.5-byte core),
- channelwise 4-bit shell layers,
- output-specific 2:4 structured sparse first classifier layer,
- 4-bit final classifier,
- FP16 biases/scales,
- exact support coding.

The resulting serialized model is **9,926 bytes** for all eight confirmatory seeds. Pack/unpack was exact at the output level (max logit difference 0). Confirmatory metrics after codec roundtrip:

- combined fidelity: **0.9636**, 95% CI **0.9516–0.9740**,
- vs dense 4-bit compiled model: **0.9735**, 95% CI **0.9641–0.9833**,
- matched-control utility: **0.9835**, 95% CI **0.9639–1.0059**.

This result also demonstrates a key allocation principle: minimizing every component independently is not optimal. The globally stable model used a *larger* core (296 B rather than 44.5 B) so that the head could be compressed while retaining whole-network robustness.

## Current working theory

The most defensible synthesis is:

> Trained neural networks can contain task-conditioned compositional redundancy. Some spans admit intrinsic simplification; additional repair can induce adaptive compensation. The remaining computation is partially redistributed and reparameterized, but under multiple operational codecs the whole network can still become substantially shorter. The attainable global code length depends on how complexity budget is allocated across the graph, not on independently minimizing each local component.

The strongest untested extension is external validity: other datasets, architectures, skip-graph topologies, and arbitrary subgraphs.
