> 歴史的な事前計画の科学部分を抜粋した資料。本文の準備・未実行という時制は当時のもの。現在の実行状態は結果報告を参照。

> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# R87R2 fixed-budget validation control census

## Question and hypothesis
Does the old eight-seed descriptive pattern—jointly fitting the two-module span removes much of local fitting's feature-error gap—remain visible in the 16-seed R87R2 cohort under its exact-reproduced numerical recipe?

Use the existing fixed 2,048-parameter three-arm control, not each method's selected minimum-budget endpoint. Primary descriptive directional prediction: joint-factorized endpoint validation NMSE is lower than local componentwise NMSE in every one of the 16 model seeds. Report exact sign counts and every seed even if the prediction fails. This new rule is not retroactively preregistered, and no confirmatory PASS may be assigned.

## Immutable source
Repository worktree ARTIFACT_ROOT/REDACTED, commit 056d69cc257a7670fb45e17f11588afa497d8881.

- scripts/reproduce/core_discovery_digits/run_confirmatory.py: blob 8759933ed2bb95014c3afc835acafa1743e40ea6; SHA256 70407ed11607c409f4b24664d5a1dc9099f494b8136526c128b6a12098fea1bb.
- results/reproduction/r87r2_fresh_sqrt_profile/FRESH_ROWS.json: blob 4abb44a7d2ee8fc22df2b9352931898b338a0404; SHA256 67ccf58d134b7a210335c77d3d634d30a93417fd6b7bd41e8dc175f7ef172301.

Frozen original primary host a, seeds 871200 through 871215, all 16; no exclusions, alternate hosts, or technical-repeat inflation. The historical 1200–1207 cohort is not pooled.

## Comparability established from source
At h=8 the local arm is two TinyRes(64,8) modules, each with two bias-free matrices, totaling 2*(64*8+8*64)=2,048 learned parameters. Joint-factorized is the same two-module shape and total parameter count. Comp is one TinyRes(64,16), totaling 64*16+16*64=2,048. All replace first two teacher blocks with fixed remaining network.

All modules receive 600 optimizer updates; local has two separate optimizers/fit calls (1,200 optimizer steps summed), joint and Comp each 600. Parameter-update exposure is equal, but forward/backward operation counts and supervision differ. Local's second fit uses teacher-produced intermediate inputs; joint uses its first student's outputs. Each arm has different initialization seeds and minibatch streams. Thus equal parameter budget is exact, equal measured compute is NOT established, and the comparison does NOT isolate objective alone, topology alone, covariate shift alone, or a causal mediation fraction.

## Restricted data handling
Hash the immutable blob and verify source without executing it. The scalar JSON container includes test fields; reading its bytes for integrity/JSON parsing is unavoidable. The analysis must immediately project only allowlisted validation fields and must not print, calculate, compare, select, tune on, or otherwise use any test metric. Do not describe this as literal absence of test-containing bytes. No datasets, test arrays, models, checkpoints, logits, or endpoint tensors are opened. Importing torch or an experiment runner is forbidden.

Allowlist per row: seed, teacher_val_acc; unique grid entry with budget=2048 and h=8: sep_nmse, comp_nmse, sep_val_acc, comp_val_acc; joint_factorized_2048: nmse, val_acc. Do not include selected_* fields. Output a validation-only projected input file with its own hash so the computational stage has no test fields.

## Fixed finite calculations
1. Abort on source/hash mismatch, non-primary host, missing/duplicate/extra seeds, absent/nonunique h=8 budget=2048 row, nonfinite metrics, or invalid [0,1] accuracy / negative NMSE. Never regenerate source results.
2. For each seed report teacher validation accuracy and L/J/C NMSE and validation accuracy. Compute fixed original utility indicators NMSE<=0.08 AND accuracy>=teacher_val_acc-0.02. These are descriptive at a fixed budget, not newly selected endpoints or a fresh quality test.
3. Compute all three paired contrasts J-L, C-L, C-J for NMSE and validation accuracy; report mean, median, min, max and negative/zero/positive counts. No bootstrap, p-values, cutpoint searches, seed weighting or exclusions.
4. Report the arithmetic NMSE decomposition (L-C)=(L-J)+(J-C), with all terms and no causal interpretation. Do not divide by tiny or negative gaps, or advertise a 'percentage explained' as a causal statistic.
5. Report exact cross-tab of original utility indicators for all 16 seeds. Do not let improved feature NMSE erase a task-quality failure. No success-driven follow-on or new training follows from this report.


## Nonduplication check
Inspected current checkpoint SCIENTIFIC_APPENDIX, original CORE_DISCOVERY_REPLICATION_DIGITS section, R87R2 saved-row audit README and implementation, and bounded searches of existing Canaria analysis .md/.py files outside source mirrors. Original documentation gives the old eight-seed mean L/J/C NMSE (0.1474/0.0639/0.0533); the R87R2 audit checks 112 grid candidates, first-passing selections and the original selected-endpoint decision, but does not compute this 16-seed equal-budget joint-control census. Technical replay checked full-object equality, not this substantive comparator decomposition. No equivalent census was found in those checked records; this is bounded evidence, not proof about every external historical artifact.

## Claim boundary / next decision
A favorable directional pattern would support consistency of the old descriptive control under this new recipe, not a new scientific sample or isolated causal mechanism. Mixed or reversed signs would qualify that interpretation while preserving original selected-budget confirmation. Neither outcome repairs historical Issue87 or shows useful whole-model compression. Use the census to decide which exact comparator must be controlled in a future untouched-confirmation design; do not fit a new variant from the result automatically.
