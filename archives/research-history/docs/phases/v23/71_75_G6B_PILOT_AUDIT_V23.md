# G6b pilot / adaptation audit trail — v23

This file records the pre-confirmatory history for the real-text causal LM phase. None of the pilot seeds are included in confirmatory inference.

## Initial pilot
Seed 3499 was reserved for pilot use. The scientific structure was fixed as a 4-block causal Transformer compiled to 2 smaller causal blocks using 512 unlabeled residual-stream calibration windows, followed by bounded repair and matched continued-training controls.

The first execution attempts exceeded the 120-second runtime limit **before producing result files**. Before any confirmatory seed was run, compute-only amendments reduced sequence/model scale and separated functional evaluation from q8 storage follow-up. These amendments did not use an observed full-pilot outcome.

The completed seed-3499 pilot then showed:
- baseline validation PPL about 19.85;
- tau0 PPL utility about 0.997;
- tau0 greedy rollout agreement about 0.605;
- shell-only tau8 PPL utility about 0.948;
- shell-only tau8 rollout agreement about 0.306.

Thus teacher-forced likelihood could be preserved while free-running dynamics diverged.

## Bounded final adaptation menu
Before any seed >=3500 was run, exactly three variants were frozen for two additional discovery seeds (3497, 3498):

1. **V1 current** — 2 compiler blocks, MLP=24, shell-only tau8 repair.
2. **V2 capacity** — 2 compiler blocks, MLP=48, shell-only tau8 repair.
3. **V3 joint-repair** — 2 compiler blocks, MLP=24, tau8 joint repair of compiler and shell at lower LR.

Selection rule:
- require parameter reduction >=35%;
- score each variant by mean across discovery seeds of `min(PPL utility, greedy rollout token agreement)` at tau8;
- ties within 0.01 select the smaller compiled model.

Observed discovery scores:
- V1: approximately 0.391 mean;
- V2: approximately 0.484 mean;
- V3: approximately 0.488 mean.

V2 and V3 differed by <0.01, so the written tie rule selected the smaller **V3 joint-repair** condition (52.28% parameter reduction). The confirmatory protocol and thresholds were then frozen before seed 3500.

No additional candidate family was introduced after confirmatory outcomes were observed. The poor pilot rollout behavior was not used to weaken the confirmatory rollout threshold.
