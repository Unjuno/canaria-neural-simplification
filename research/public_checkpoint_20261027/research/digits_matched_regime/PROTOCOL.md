> 歴史的な事前計画の科学部分を抜粋した資料。本文の準備・未実行という時制は当時のもの。現在の実行状態は結果報告を参照。

> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Matched local-versus-joint control: preregistration draft


## Decision in one paragraph

The next informative intervention is a paired **training-regime** contrast: identical two-map student parameters and minibatches, equal per-parameter update counts, local teacher-forced fitting versus joint endpoint fitting. Single-map Comp remains a separate architecture comparator. The current artifacts do not establish an untouched real-data confirmation set for this digits claim. A new-seed experiment on the old split could test robustness to new optimization randomness conditional on reused data; it must not be sold as fresh-data validation. Therefore the design below is executable in principle but remains gated, not a direction to restart fits now.

## Source and motivation

- Read-only source/reference commit: `056d69cc257a7670fb45e17f11588afa497d8881`.
- Source runner: `scripts/reproduce/core_discovery_digits/run_confirmatory.py`, blob `8759933ed2bb95014c3afc835acafa1743e40ea6`, SHA256 `70407ed11607c409f4b24664d5a1dc9099f494b8136526c128b6a12098fea1bb`.
- The existing `run()` is NOT suitable for direct reuse: it uses different arm RNG streams, executes an entire grid, and reads test outcomes. A new restricted runner must reuse only reviewed definitions and teacher recipe; do not invoke the historical suite or modify its seed admission list.
- Motivating descriptive result: `a32ffbb9bc79711a8f2849acd2b7766860d254b7`, `research/r87r2_equal_budget_validation/REPORT.md`. At 2,048 weights the old 16-seed NMSE order was C<J<L in all cases, but Comp task accuracy beat J in only8, tied5 and lost3. This is design input, not confirmation evidence for the new design.
- Retain `portable_v2_sqrt64`, pinned Python3.13.5/torch2.10.0+cpu/numpy2.3.5/sklearn1.8.0/scipy1.17.0, one intra/inter-op thread, deterministic algorithms, AVX2, MKL COMPATIBLE, mkldnn disabled. Porting/scoped-sqrt safety must be reviewed; no global threaded monkeypatch.

## Hypothesis and estimand

H1: At the fixed 2,048 learned-weight budget, replacing local teacher-forced training with joint endpoint training reduces final validation span NMSE under matched initialization, examples and parameter-update exposure, conditional on the original digits split and teacher recipe.

The estimand is the total effect of the specified training regime, not objective alone. Changing local losses to the endpoint loss necessarily changes the intermediate input distribution and cross-map gradient path. Even after matching nuisance factors, J−L cannot identify these mechanisms individually. C−J additionally changes topology/bottleneck allocation and is descriptive only. No budget selection, universal composition law, CIFAR rescue or deployment claim is tested.

## Fixed arms and matching

Use the original four-block64-dimensional teacher, first two blocks, frozen stem/suffix/head, and original training1077/validation270 IDs. Never load the test450 arrays in the restricted runner.

For each admitted teacher-seed ID:
1. Train one teacher using the unchanged60-epoch recipe, then freeze and hash it. Produce fit-only teacher intermediates a0,a1,a2. Generate validation intermediates only after all replacement arms have finished, to prevent checkpoint selection.
2. Initialize two bias-free TinyRes(64,8) maps once. Serialize the initial tensors and deep-clone them into L and J; exact tensor hashes must match before optimizer construction. Their2,048 parameter elements are in the same named order.
3. L: one AdamW optimizer over both maps, lr0.008, weight_decay0.00001 and the explicitly recorded inherited AdamW defaults. At update t use the shared index vector and the loss MSE(r1(a0),a1)+MSE(r2(a1),a2). The sum, not average, preserves each disjoint map's local gradient scale. One backward/step updates every map600 times. No student-output substitution in L.
4. J: same optimizer configuration and initial empty state, same indices,600 steps, MSE(r2(r1(a0)),a2). Both maps are updated each step. No local auxiliary term or detachment.
5. C: one TinyRes(64,16),2,048 weights, endpoint MSE, same600 index vectors and optimizer settings. Different tensor shapes prevent identical initialization to L/J; initialize from a separately domain-separated RNG and label C architecture-comparison only.
6. Precompute and hash a600×128 with-replacement fit-index schedule using a dedicated RNG before any arm starts. Each arm consumes exactly that schedule without advancing it; global RNG is not a scheduler. Use distinct domain-separated teacher, pair-init, C-init and index-stream seed derivations from a frozen16-ID manifest. No arm can mutate inputs/teacher/other-arm state.
7. Per parameter:600 optimizer steps and76,800 sampled example exposures. Same step count is NOT equal FLOPs or wall time. Record loss-component calls, processed examples, parameter updates and arm elapsed/RSS; report compute differences rather than pretending to equate them. Do not normalize conclusions by unmeasured compute.

This is a new matched regimen, not an exact rerun of historical L with its two separately scheduled fit calls. A mathematical/source preflight must establish disjoint-gradient equivalence of summed local losses under explicit AdamW defaults before execution. Numerical equivalence is a future preflight obligation, not something verified by this design task.

## Cohort, exposure census and holdout boundary


Verified from the pinned protocols/status and current checkpoint:
- Digits original train1077/validation270/test450 split is reused. Original seeds1200–1207, R87R2 seeds871200–871215 and known preflight1202 have been executed. Technical replay adds no independent seeds. Fixed test450 has already produced reported outcomes; withholding it from this runner does not make it pristine again.
- R87R2 source fixes split RNGs1234/5678. Repartitioning any of these1797 examples does not yield study-unexposed images. A split change is a new adaptive design, not recovery of an untouched test.
- Phase4 California Housing Stage B and its test results have already been reported; it is a different regression task, not an unused digits confirmation set.
- CIFAR original fit/calibration/validation and official test are study-exposed; official test was unblinded once. Later calibration diagnostics are seen. They cannot validate this mechanism afresh.
- Candidate documents say Phase3C's outer diabetes test was not evaluated, but no teacher met its locked quality/stability gate. This task has not independently proven its full custody history, and a different task's gated holdout is not a reusable digits confirmation route. Do not open it, bypass its teacher gate, or revive that line.
- The completed synthetic-rank experiment concerns a different analytical mechanism. Future independent draws from a newly frozen generator could be genuinely new draws, but would validate that declared synthetic distribution only. They are not untouched real-data evidence for the present digits claim and are not proposed as a substitute success headline.


## Outcomes and locked analysis proposal

Single final checkpoint at step600 only; no best epoch, widening, threshold tuning, additional seeds or retry based on results. Per seed retain raw final validation counts out of270, teacher counts, endpoint SSE and teacher centered sum of squares, NMSE, every arm's failures and identities. Do not use test metrics.

Primary: paired log2(max(NMSE_J,1e-12)/max(NMSE_L,1e-12)) averaged across all16. Proposed paired-seed percentile95% bootstrap100,000 draws with a separately frozen analysis RNG: support if upper endpoint<0, contrary evidence if lower endpoint>0, otherwise uncertain. Report every seed and mean/median/range/sign census. Bootstrap is conditional on this reused split and is not an image-level generalization interval.

Task guardrail: mean accuracy_J−accuracy_L; support noninferiority only if lower95%>−0.02, fail if upper95%<−0.02, otherwise uncertain. A feature benefit without this gate is feature-only, not quality-preserving improvement. Report original fixed-budget quality counts (NMSE≤0.08 and accuracy≥teacher−0.02) for every arm without filtering teachers/seeds or replacing this primary contrast.

C comparisons have no confirmatory pass label and cannot rescue an unsuccessful J−L primary. No pooled dataset claims, multiplicity-driven selection, causal percentage decomposition or use of old16 outcomes in the new16's inference.

## Resource budget and failure census

Proposed cap, not measured feasibility: one local container, at most two sequentially isolated single-thread workers,≤4GiB RSS per worker,≤3 hours elapsed overall,≤10 minutes per seed,≤1GiB new evidence. This is substantially smaller than the prior full grid but its actual runtime is not yet verified. Stop earlier at the existing common deadline **2026-10-29 06:50 UTC**, never extend it. No CI, paid API, external service, GPU purchase or user-PC work.



## Design-task verification limits

Inspected source text, fixed protocols/status, equal-budget report and checkpoint appendix only. No ML imports or model execution. Local document creation is the sole modification. Resource cap is proposed; seed novelty, implementation correctness, runtime feasibility and untouched-source availability are not claimed verified.

## Preparation lock (2026-10-27, supersedes draft choices only)



Explicit AdamW parameters: lr=0.008, weight_decay=0.00001, betas=(0.9,0.999), eps=1e-8, amsgrad=False, maximize=False, foreach=False, capturable=False, differentiable=False, fused=False. Teacher recipe remains exactly the pinned train_model function and its defaults; original constructor sets teacher seed. Local regime uses one optimizer with sum of disjoint local losses. Clone two-map initial state to J. Both600 steps. Comp600 steps on same schedule. One scientific process per seed, maximum2 processes, each single-thread. Scoped sqrt override is isolated to a single Python thread and restored even on exceptions; no multithreaded optimizer calls.

Quality condition: validation NMSE <=0.08 and exact correct/270 >=teacher_correct/270−0.02. NMSE denominator preserves original float32 centered target MSE +1e-12, with raw SSE/count and centered sum retained for audit. Primary paired-seed percentile95 bootstrap100000 resamples, one shared index matrix (271027042), arithmetic mean log2(J/L), PASS upper<0, FAIL lower>0, otherwise UNCERTAIN. Accuracy NI lower>−.02 PASS, upper<−.02 FAIL, else UNCERTAIN. All16 required; missing/nonfinite/invalid=>INCOMPLETE, not filtered. Report C descriptively only.

Do not load or evaluate test-partition feature/target arrays; obtaining original split indices uses full dataset labels for stratification as in the original recipe. Full raw sklearn digits bytes are not an untouched-data boundary. Model evaluation on validation starts only after all three student fits finish within that seed. No checkpoint/budget selection.