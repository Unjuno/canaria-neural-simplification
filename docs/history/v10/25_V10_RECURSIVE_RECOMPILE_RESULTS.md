# v10 Recursive Recompile Experiments

## 0. Status
All experiments in this document are **pilot-only** and use seeds 900–902. They do not consume or inspect confirmatory Phase-A Canary/simplification outcomes for seeds 1000–1015.

The goal was to test the strongest version of the recursive-compiler hypothesis raised after v9:

> After a full-span compile has recovered utility through shell adaptation, can the apparently more complicated shell itself be compiled again while preserving matched-control utility?

## 1. Experiment A — whole post-shell recompile
Raw: `raw_experiments/canary_core_recursive_recompile_pilot_v10/`
Script: `scripts/run_canary_recursive_recompile_pilot_v10.py`

Stage 1 reproduces the v9 full-core compile followed by post-shell-only 8-epoch repair. Stage 2 replaces the entire repaired post-shell (`b_out + MLP head`) by a single ridge-linear map from compiled-core output to logits. No additional repair is allowed. Stage 3 further attempts front/pre-shell collapse.

### Result
- Stage-1 Conv1: mean utility vs matched control = 0.9815, pass rate 3/3.
- Stage-1 Conv3: mean utility = 0.9830, pass rate 3/3.
- Stage-2 single linear post-shell:
  - Conv1 mean utility = 0.7504, pass rate 0/3.
  - Conv3 mean utility = 0.7626, pass rate 0/3.
- The failed Stage-2 models were very small (5,866–6,378 parameters; fixed-FP32 reduction 82.1–83.6% vs baseline), but the utility loss is decisive under the 0.95 floor.
- Further Conv5/Conv7 front collapse reduced utility more strongly and is not viable.

Interpretation: the post-shell computation created/used after first compile is not representable by a single linear boundary map under the current task distribution.

## 2. Experiment B — post-shell latent-width sweep
Raw: `raw_experiments/canary_core_recursive_posthead_width_pilot_v10/`
Script: `scripts/run_canary_recursive_posthead_width_pilot_v10.py`

The single-linear failure could be caused by grammar undercapacity. The post-shell was therefore replaced by a one-hidden-layer MLP head with widths 8/16/32/48, trained only to match teacher logits on clean + fixed augmented training inputs. Labels are not used in this second compile.

### Result
No width passed the 0.95 matched-control utility floor across the three pilot seeds.

Conv3 branch:
- width 8: mean utility 0.5598
- width 16: 0.7154
- width 32: 0.8827
- width 48: 0.9178

Conv1 branch:
- width 8: 0.4506
- width 16: 0.6520
- width 32: 0.8602
- width 48: 0.8706

Width 48 is already close to the Stage-1 total parameter count once the retained front/core parameters are included, so simply increasing latent width does not provide evidence for a second strong compression step.

## 3. Experiment C — boundary-local recursive compile
Raw: `raw_experiments/canary_core_recursive_head_boundary_pilot_v10/`
Script: `scripts/run_canary_recursive_head_boundary_pilot_v10.py`

Rather than collapsing `b_out + head` together, the boundary was localized:
1. keep the repaired `b_out`, replace only the MLP head with a ridge-linear head;
2. then replace `b_out` Conv3 by Conv1+ReLU and refit the linear head;
3. remove `b_out` entirely as a boundary control.

### Result
The strongest compact result was the Conv3 branch with `b_out=Conv1+ReLU + linear head`:
- mean utility = 0.9513
- minimum utility across seeds = 0.9443
- pass rate = 1/3
- parameters = 6,450
- fixed-FP32 reduction = 81.94%
- q8 entropy-code reduction = 83.30%

This is close to the utility boundary in the mean but is not seed-stable. Removing `b_out` entirely falls to mean utility 0.9289.

Interpretation: a small nonlinear boundary operator carries material function that cannot simply be deleted. The data are more consistent with a **recursive compression frontier / fixed-point neighborhood** than with unlimited repeated collapse.

## 4. Experiment D — recursive compact-model repair phase
Raw: `raw_experiments/canary_core_recursive_repair_phase_pilot_v10/`
Script: `scripts/run_canary_recursive_repair_phase_pilot_v10.py`

The compact second-stage model (`compiled core + Conv1 b_out + linear head`) was allowed additional supervised repair for tau={0,1,2,4,8}, training only `b_out + head`. The matched no-compile control received the same extra tau after its original 8 continuation epochs.

### Result
Absolute augmented accuracy increased with repair, but the matched control improved faster.

Conv3:
- tau 0: mean compact accuracy 0.6822, mean utility 0.9505
- tau 4: accuracy 0.7348, utility 0.8952
- tau 8: accuracy 0.7533, utility 0.8709

Conv1:
- tau 0: utility 0.9229
- tau 8: utility 0.8516

Thus extra training does **not** rescue the strongest 82%-compressed recursive model relative to a matched continued-training control.

## 5. Current interpretation

### Supported by this pilot
1. First-stage full-core compile + post-shell repair is reproducibly viable on seeds 900–902.
2. The shell cannot be recursively collapsed arbitrarily under the tested grammar.
3. There is evidence for a compact nonlinear boundary requirement: `b_out` removal is harmful, while retaining/reducing it is substantially better.
4. The strong form `compile -> repair -> compile -> repair -> ...` with monotone utility-preserving code reduction is not supported.

### Not established
1. The post-shell is mathematically irreducible.
2. A richer symbolic/gating/piecewise grammar could not compress it further.
3. The same fixed-point behavior occurs outside residual-8 digits CNNs.
4. The observed frontier is independent of the selected task distribution / augmentations.

## 6. Updated hypothesis
The leading local model is now:

> `net simplification + computational redistribution` occurs at the first compile, but redistribution terminates near a task-conditioned nonlinear boundary complexity floor under the current grammar.

This is more restrictive than the earlier unlimited recursive-compiler hypothesis and is directly falsifiable by richer grammar or cross-task replication.

## 7. Next experiment priority
Do not continue expanding pilot grammar indefinitely before the preregistered Phase A. The next high-value step is the locked confirmatory residual-8 Phase A on seeds 1000–1007 with the frozen v9 grammar and tau={0,1,2,4,8}, preserving the Stage-1 hash-before-Canary barrier.
