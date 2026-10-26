# Local finishing verification, 2026-10-26

Base main: c7fd4fb701064c9aeef5cc8109231d884e5258aa.

These are local container verification records, not a release, new confirmatory cohort, independent external replication, or clearance of Issue87. Plans were announced to the owner before execution. GitHub CI was not used. Remote publication is pending authenticated write access.

## Synthetic rank same-code rerun

Source PR104: ffc4f62034743b2d57c86437958a2e4b14aa52db. The original fixed 64 seeds per regime were rerun once, unmodified. All 192 rows completed. Cost ratios were 1.0 (independent), 0.5 (shared), and 0.0 (inverse). An independent implementation rederived ranks/costs/threshold decisions from saved spectra and matched archived numerical fields. Matrices and condition numbers were not independently regenerated. This concerns a synthetic linear oracle, not practical neural-network acceleration.

## SmallViT objective-control saved-output audit

Source PR105: 50a73cd44a05a62e28c94a9e67c84f7b7e9e86cf. The evaluation-only path reproduced DECISION.json and FRESH_ROWS.json byte-for-byte. An additional persistence audit verified 16 records, 54 manifest hashes, 121 logits arrays and reconstructed accuracies, and 60 state hash groups. Fifteen seeds were eligible. The scientific decision remains UNCERTAIN: the NMSE improvement gate is not established, while the test noninferiority gate passes.

NMSE validation uses saved MSE and denominator, not regenerated hidden activations. Saved freeze certificates do not independently prove historical chronology of test access. No retraining or new seeds were performed.

See the source receipts, audit implementations, raw results and environment records in each subdirectory. Absolute scratch paths in original receipts record execution provenance; preserved relative copies are included here.

## Phase4 grid-to-decision independent audit

Source PR101: 056d69cc257a7670fb45e17f11588afa497d8881. Reconstructed 144 validation predicates and 16 first-passing selections across all eight seeds; 978 checks passed. Recomputed decision matches every archived value, PHASE4_CONFIRMATORY_PASS. Geometric budget ratio 0.4760285231 (transformed 95% interval 0.3982532982–0.5615479995), lower in 8/8 seeds. Teacher test R2 mean 0.7950600304; selected paired test-R2 difference +0.0068578051. Fixed 100000 paired bootstrap draws, seed 1355817528.

Additional source-history inspection verified protocol bytes, runner blob and Stage-A -> lock -> launch -> persisted-result ancestry. Git history supports recorded ordering, not independent proof of real execution timing. Second-host raw records remain unavailable. This is saved-metric recomputation, not fresh training or hidden-prediction reconstruction.

## R87R2 saved candidate-to-decision audit

Source PR101: 056d69cc257a7670fb45e17f11588afa497d8881. Verified all 112 candidates, 224 validation flags and 32 first-passing endpoints. Independent fixed paired bootstrap and original evaluator match every archived decision field. Geometric budget ratio 0.5316098878, 95% CI [0.4859031281, 0.5946035575]; selected test accuracy difference +0.0037500113, CI [0.0002777874, 0.0075000116]. Both locked gates PASS. Fifteen of sixteen seeds have lower composed budget.

One initial audit-only full-object snapshot equality assertion was too strict for the source copying order; its failure is retained and correction documented without changing scientific thresholds. This verifies saved scalar arithmetic, not raw weights/predictions, historical launch prerequisites or cross-host parity. Historical eight-seed Issue87 evidence is separate.

## Preservation audit

An additional read-only review at commit 4e53c895bd7edb2c1c636d895165e97f3616bf65 completed 219 integrity/provenance checks with zero failures. Recorded file hashes and pinned source copies matched. Its receipt is PACKAGE_REVIEW.json. Original scripts are retained unchanged: the rank auditor expects the original repository scratch path, and the original R87R2 audit README references its execution scratch directory. These are portability limitations, not evidence of scientific failure. Reconstruct the referenced source tree/paths or adapt a separately versioned wrapper before rerunning elsewhere; do not silently edit preserved source bytes.

## R87R2 actual training technical reproduction

The pinned PR101 local suite executed the known1202 bridge followed by all16 fixed cohort seeds with workers2. All17 processes exited0; no retries were logged. Entire actual outcome objects and the decision exactly equal the archived primary host. An independent read-only audit passed103 checks, including87 original suite hashes, pinned source hashes, numerical environment and AVX2/MKL/profile gates. See r87r2_training/INDEPENDENT_REPRODUCTION_AUDIT.json.

Python3.13.5, CPU PyTorch2.10.0, NumPy2.3.5 and all pinned dependencies were used in an isolated environment. Host Debian13 differs from the original Ubuntu24.04 workflow. Missing ensurepip was handled via uv isolated bootstrap; lscpu topology was unavailable and /proc/cpuinfo was retained. These are documented host/setup differences, not changes to numerical settings or scientific code. This is fixed-cohort technical reproduction, not fresh independent seeds, historical8-seed recovery, external replication, or independent re-establishment of the old three-host launch prerequisite. No GitHub CI was used.

## SmallViT hidden-test saved-record closure replay

Pinned PR105 50a73cd44a05a62e28c94a9e67c84f7b7e9e86cf. Existing locked evaluator and stronger closure auditor both exited0; parsed outputs and bytes match archived DECISION and CLOSURE_AUDIT exactly. Sixteen attempted, thirteen eligible, exclusions9202/9203/9213 by baseline validation accuracy below0.95. All13 eligible seeds selected a smaller composed budget; geometric ratio0.511765014826. Seven mutation checks and mocked selection/test-control-flow checks passed. No training or new seeds.

The test dataset exists in the same process; code-path evaluation order is checked, not external access control or historical logs. Compiler holdout is a disjoint subset of teacher training data. All componentwise selections hit the two-block grid minimum, so the near-half ratio is candidate-grammar-specific, not a universal complexity law.

## Phase4 actual training technical reproduction

Pinned full source origin/phase4 f7d23c9dad2788e64b94f2a268fe3e0b1084d77c, with protocol/wrapper/evaluator/expected rows equal the PR101 candidate. Known2899 preflight passed before recorded2900–2907; nine exits0, no logged retries. All eight entire records and the final decision exactly match archived primary evidence. Independent read-only audit154/154 checks passed, including all62 sealed files, dataset X/y/split hashes recomputed from the downloaded offline cache and bootstrap statistics. No new independent seeds or CI.

PR101 lacks the pinned base runner at scripts/phase3/regression_external_validity/run_seed.py. This execution used its exact original blob8dd3ddefb59ba250f5deca2d98d04fbff0055748 from the full Phase4 research branch; therefore this does not claim PR101 is standalone runnable before the packaging repair. Local result remains a technical reproduction, not a new scientific cohort or external peer review.

## Candidate-only packaging repair

Separate local branch research/candidate-repro-packaging-20261026, commit3252e1baa13241817b0e32969ab0468f38abcf0f, based onPR101056d69cc257a7670fb45e17f11588afa497d8881. Two additions only: byte-identical locked Phase4 base runner and reproduction documentation. Existing scientific files/workflows untouched. Independent review approves the narrow packaging patch; compile/import with blocked network and18 tests plus repository audit pass. Patch and receipts preserved under candidate_packaging. No push, merge, release or claim promotion is implied. Candidate-level full training has not been rerun after this packaging-only fix; identical original base was used for the separately audited Phase4 run.
