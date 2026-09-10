# R87: tested numerical repair and fresh confirmation

## Result

The new `portable_v2_sqrt64` recipe fixes the observed numerical mismatch on the tested hosts and supports a separate fresh confirmation. This is **not recovery of the historical archive**. Original scientific source, original recorded outcomes and original comparison tolerances are unchanged.

R87D5 showed that forced ATen AVX2, MKL COMPATIBLE, single-thread kernels, disabled oneDNN and deterministic algorithms alone were insufficient in the original Intel/AMD comparison. R87D6 common-input replay localized an additional difference to float32 square root inside AdamW, after identical first and second moments. R87D7 evaluates only float32 Tensor.sqrt calls during AdamW.step in float64 and casts back to float32. Model/moment storage, teacher and replacement losses, data, fitting schedules and scientific gates are unchanged. The override is restored after the step, including on exceptions. This is an isolated-process research implementation, not a thread-safe general optimizer or a compiled/GPU guarantee.

## R87R2 fresh scientific result

The protocol was fixed before outcomes. Primary scientific host A, seeds871200-871215; host B and local reruns are technical replicas, not additional independent seeds. The earlier conditional R87R1 was not run after D5 failed; its seeds were not reused.

| Quantity | Result | Paired bootstrap95% | Decision |
|---|---:|---|---|
| Mean selected component-wise replacement budget |3392 parameters|descriptive|—|
| Mean selected composed replacement budget |1824 parameters|descriptive|—|
| Geometric composed/component-wise budget ratio |0.5316098878|[0.4859031281,0.5946035575]|PASS: upper log-ratio interval below zero|
| Selected composed minus component-wise test accuracy |+0.375001pp|[+0.027779,+0.750001]pp|PASS: lower above-2pp|
| Composed selected budget smaller |15/16; remaining seed ties|descriptive, not population certainty|—|

Decision: **R87R2_CONFIRMATORY_PASS**. Every seed has both selected endpoints. Paired model-seed bootstrap100000, RNG87122026. Test endpoints are evaluated under the original validation-only budget selection rule. This is a reused fixed test split, not a new independently sampled dataset.

## Additional technical verification

Complete source/raw artifacts from both GitHub AMD EPYC9V74 hosts were recovered. A further complete rerun on an Intel Xeon Platinum8573C matched all16 scientific outcome objects, teacher/replacement state hashes, input/init/RNG ledgers and all saved numeric checkpoint arrays exactly. No outcome tolerance was widened. Original evaluator recomputation and a separate implementation of primary bootstrap endpoints agree exactly.

The complete local closure audit covers108 primary/repeat records and609 original-manifest files: D5 two remote hosts40, D7 two remote hosts20, and R2 three directly accessible hosts48. Preflight bridges are separate. These are NOT108 independent seeds: R87R2 has16 new model seeds and one dataset. Remote CI rechecks the remote raw files and compares their scientific-byte fingerprint against the independently recorded Intel receipt; that receipt comparison is not another live Intel execution inside CI.

Earlier Intel Xeon E5-2673v4 local execution receipts remain in LOCAL_EXECUTION_RECEIPTS.json. Their temporary raw arrays did not survive the session reset. The present closure does not pretend to re-audit those unavailable arrays; Intel/AMD agreement is independently supported by the new complete Platinum8573C rerun in the companion session bundle.

## Conditions

Python3.13.5, torch2.10.0+cpu, NumPy2.3.5, scikit-learn1.8.0, SciPy1.17.0. Effective ATen AVX2 and MKL CNR:COMPATIBLE are checked. Torch intra/inter-op threads1; teacher batch64/60epochs, replacement batch128/600updates. Four independent worker processes in the extra Intel rerun, each with one-thread kernels. CPU/build/OS data are retained. No controlled clock, latency, memory or energy benchmark is claimed.

## Boundaries and failed attempts

Old archived known-cohort means were3584/1728 and paired geometric ratio0.4823393150. D7 happens to have the same two means, but its paired ratio0.4926924861 and per-seed/test values differ. Matching aggregate means does not recover the old experiment. Its producing environment and original raw provenance remain unrecovered; Issues87/13 are not silently closed.

The original D6 replay failed during metadata lookup after saving numeric arrays; it is not a completed successful report. Metadata was corrected without changing old arrays. The extra Intel preflight initially lacked the unchanged core runner because a slim source export was assembled as a full checkout; it failed before training. The base source was restored under its recorded SHA256 and the full suite then completed. Failed-attempt records are preserved in the handoff.

No main claim promotion or automatic release occurs. A scoped new statement may be reviewed: the fixed recipe yields smaller selected composed replacements on average in a fresh16-model digits cohort while retaining the original test non-inferiority gate, and identical computation on the directly tested Intel/AMD hosts. This is not universal compression, a minimum-size theorem, independent dataset replication, general bitwise portability or measured hardware resource saving.
