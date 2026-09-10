# R87 diagnostics: numerical execution-path sensitivity

Evidence: known-cohort diagnostics, NOT a new independent confirmatory experiment. The original scientific runner, archive and thresholds remain unchanged. Issue87 and the main announcement blocker remain open.

## R87D1 — controlled dispatch intervention

Protocol lock: `00969cc660e2ac628f9d6220ad9481d62f195f92`. Scientific runner blob: `8759933ed2bb95014c3afc835acafa1743e40ea6`. Instrumentation SHA256: `24c849420debcd3f225f13362127ecba7949237041cef7ad61e3209588e61912`.

On the local AMD EPYC9V74 host, Python3.13.5 / torch2.10.0+cpu / NumPy2.3.5 / sklearn1.8.0, the paired intervention changes only `ATEN_CPU_CAPABILITY`: unset (reported AVX512) versus `avx2` (reported AVX2). Eight known seeds1200–1207, plus independent-process repeats of both cells on1202/1205. One Torch thread. No timing/energy claim or fixed clock assumption.

Data tensors, initial parameters, complete tracked random shuffle/minibatch streams and first teacher logits were identical in all8 pairs. First parameter gradients differed in all8 pairs (maximum absolute differences7.45e-9 to2.05e-8). Four pairs changed a selected budget:

| Seed | Native: component/composed | AVX2: component/composed |
|---|---|---|
|1200|3072 /1536|3072 /1536|
|1201|3072 /2048|3072 /2048|
|1202|3072 /1536|4096 /1536|
|1203|3072 /1536|3072 /1536|
|1204|3072 /2048|3072 /1536|
|1205|3072 /2048|3072 /1536|
|1206|3072 /1536|4096 /2048|
|1207|3072 /2048|3072 /2048|

Means: native3072/1792, paired geometric ratio0.5773502692; AVX23328/1728, ratio0.5183073248. Composition was smaller in8/8 in both cells. All4 within-cell repeated full outputs and trace ledgers matched exactly. These are repeats of the same8 models, not40 independent models/datasets.

For1202, identical initial parameters/first logits led to a first-gradient max difference1.4901161193847656e-8, first-update max parameter difference3.7569552659988403e-6, and final teacher max parameter difference0.1044861823. At budget3072 the NMSE criterion passed both ways, but the unchanged teacher-relative validation utility criterion crossed its threshold. This records measured amplification, not a universal instability theorem.

The GitHub paired-host negative control (run34493859094) was natively AVX2. Explicit AVX2 therefore did not change the actual capability: all8 paired full outcomes and saved numeric checkpoints matched exactly, and4 repeats matched. Both cells reproduced the prior GitHub repeat magnitudes3456/1728 and0.5107322488, NOT the original archive3584/1728 and0.4823393150. See `github/SUMMARY.json` and raw records.

## R87D2 — first-batch operator replay, post hoc

Using known1202 and a common saved native first-batch input/parameter/upstream-gradient archive,20 captured operations were replayed on both local dispatch paths. Linear, LayerNorm and GELU operator outputs and input/parameter gradients agreed in this probe. Cross-entropy's logit gradient differed by at most9.3132257461547856e-10. Decomposing the loss found a log-softmax forward difference up to4.76837158203125e-7 and backward difference up to9.3132257461547856e-10; NLL output-gradient was exact. This isolates a present numerical path; it does not prove that no other operator can differ on other inputs.

Executed diagnostic source SHA256 `90db00fe50997fcf25a033457d2fed45796ab1289bc63a678662d2febf12467e`; loss decomposition source `7c53155e759b626b73085eb79abdf24797d0598183ff3a2a84d8f59c648cbc80`. Exact executed local source, common NPZ arrays and full local traces are in the companion session archive. The portable root-resolution form reproduces all captured/replayed native arrays exactly.

## R87D3 — same-host package substitution, prospectively locked

Protocol `../r87_version_bridge/PROTOCOL.json`, lock `f2ec479db88ed551f373153f1597aa4a19ce8afe`; run34494605928. On one GitHub host, change only the torch CPU package2.10.0 versus2.13.0 while keeping Python3.11, NumPy2.4.6, sklearn1.9.0 and other installed Python dependencies fixed. Both use explicit AVX2. Exact original/instrumented bridges precede all eight paired known seeds and four repeats.

All8 full outcome JSON objects and saved numeric stages match across torch versions. All data/init/random controls and non-torch package versions match. All4 repeats match. Decision: `NO_PACKAGE_ENDPOINT_DIFFERENCE_OBSERVED`. Both cohorts are3456/1728 and0.5107322488. Thus a torch-version-only explanation is not supported in this controlled setting; this is not a proof that versions never matter. See `../r87_version_bridge/github/SUMMARY.json`.

## R87D4 — limited numerical mitigation, post hoc

Before execution, fixed known1202/1205 only in Issue87 comment5621082695. Evaluate only teacher cross-entropy in float64, backpropagating through the cast to float32 logits; all model parameter storage and replacement MSE fits remain original float32. This is a different numeric recipe, not a patch silently applied to original evidence.

Across nativeAVX512 versusAVX2, both seeds selected3072/1536. Initial parameter gradients and checkpoints through update170 matched for both. Seed1202's final teacher and full outcome JSON matched; seed1205 retained a final-teacher max difference5.960464477539063e-8 and nonidentical full outcome JSON. Therefore endpoint agreement improved in this two-seed diagnostic, but exact cross-dispatch reproducibility is NOT established. Executed wrapper SHA256 `4f4adea5af91d7032a85b05b12eb21faed82bfe09342a14b39a55727f3c56c79`.

## What is resolved and what is not

A controlled current source of endpoint sensitivity has been identified: ATen dispatch can change low-order loss/gradient arithmetic and the subsequent optimized endpoints, despite identical seeds/data/initial parameters. On a fixed tested AVX2 host, torch2.10 versus2.13 alone did not change this computation. This narrows the causal investigation.

It does NOT recover the historical producing environment or establish exact archived numerical reproduction. LocalAVX2 and GitHubAVX2 can still differ across hosts even with the same recordedtorch2.10 build; ATen's reported capability does not constrain every external library's dispatch. That cross-host comparison is confounded by other environment/host differences and is not a unique MKL/OS diagnosis. Original producing-source/data/environment recovery and a separately locked portable numerical-profile study remain open. No main claim promotion, tolerance widening, seed replacement or automatic announcement.

## Evidence locations

GitHub: full R87D1/R87D3 JSON/NPZ, protocols, source, build metadata and manifests. Companion `canaria_r87_diagnostics_2026-09-11.zip`: complete local R87D1/R87D2/R87D4 records, exact executed sources, portable sources and independent audits. Local results are not represented as GitHub-executed outcomes. All intervals describe already-observed seed-cohort repeats, not independent external validity.
