# V11 Phase B — Shell Locus × Capacity × Replacement Severity

Status: Phase-B shell-locus protocol on new seeds 1100–1107 completed. Capacity follow-ups are explicitly exploratory/post-hoc and do not alter the preregistered Phase-B decision.

## Primary Phase-B intervention

Fixed full-span replacement, repair budgets tau={0,1,2,4,8}, matched no-compile continuation. New residual-8 seeds 1100–1107 were all baseline eligible (clean 0.960–0.971). Repair modes: frozen, pre_only, b_out_only, head_only, post_only, full_shell.

### Conv1 preregistered decision

- tau=8 post_only - pre_only mean utility difference = +0.2715, 95% seed-bootstrap CI [0.2425, 0.2999]. D_post_locus = PASS.
- tau=8 post_only pass rate (U>=0.95) = 0.375, 95% bootstrap CI approximately [0.122, 0.750]. D_adaptive_recovery = UNCERTAIN.
- tau=8 b_out_only - head_only = -0.2466, CI [-0.2647, -0.2285]. b_out localization is not supported; head adaptation is much stronger.
- full_shell utility: tau1 0.9574, tau2 0.9669, tau4 0.9626, tau8 0.9616. Pass rates: 0.75, 0.625, 0.625, 0.625.

## Equal-capacity challenge

The primary locus comparison is confounded by trainable capacity: pre_only=664 trainable params, b_out_only=584, head_only=25,114, post_only=25,698.

A masked-head follow-up constrained the head to the first k hidden units. Effective trainable parameter counts were 533, 1,056, 2,102, 4,194, 8,378, and 25,114 for k=1,2,4,8,16,48.

At tau=8:
- k=1 (533 params): U=0.6231, pass 0/8.
- pre_only (664 params): U=0.6749.
- b_out_only (584 params): U=0.6698.
- full head k=48 (25,114 params): U=0.9135.

Paired equal-capacity comparison: head-k1 was not stronger than small pre/b_out shells; at tau=8 it was lower than pre by -0.0517 (95% CI [-0.0789,-0.0242]) and lower than b_out by -0.0467 (CI [-0.0708,-0.0246]). Therefore the large post-shell effect cannot be attributed to locus alone.

## Capacity distribution within the head

Selecting the baseline-importance top-k hidden units improves partial-head repair but does not eliminate the capacity requirement. At tau=2:
- top-16 (8,378 params): U=0.8994
- top-24 (12,562): U=0.9222
- top-32 (16,746): U=0.9407
- top-40 (20,930): U=0.9431
- all-48 (25,114): U=0.9554, pass 6/8

Thus the adaptive compensation is distributed across most of the classifier head rather than concentrated in a very small subset of hidden units.

## Replacement severity control: Conv3

The same 8 new seeds were rerun with a less aggressive fixed full-span Conv3 replacement.

- tau=0 frozen U=0.9839, pass 7/8: Conv3 is already intrinsically close to the original full span.
- full_shell U: tau1 0.9873, tau2 0.9943, tau4 1.0002, tau8 0.9915; pass rate=1.0 from tau2 onward.
- post_only U: tau2 0.9792 (pass 8/8), tau8 0.9670 (pass 5/8).
- pre_only U: tau2 0.8579, tau8 0.7071.
- post_only - pre_only grows from +0.0531 at tau1 to +0.2600 at tau8.

Conv3 full-shell utility exceeds Conv1 full-shell utility by about +0.027 at tau2 (95% paired CI [0.005,0.045]) and +0.030 at tau8 ([0.015,0.044]). Hence shell-capacity requirements interact with replacement severity.

## Interpretation

The data reject a one-dimensional story. There are at least three components:

1. intrinsic replaceability: less aggressive Conv3 can replace the full 8-block core with little immediate utility loss;
2. adaptive compensation: trainable downstream shell markedly improves aggressive replacements;
3. shell-capacity dependence: for Conv1, equal-capacity post-shell controls do not outperform equally small pre/b_out controls; recovery appears only when a large fraction of the classifier head is allowed to adapt.

Therefore `adaptive simplification` is partly a real composition phenomenon but is also substantially enabled by excess downstream capacity. The strongest current model is not pure simplification and not pure relocation; it is a phase diagram in replacement complexity × repair time × available shell capacity.

## H/T/D/C/U

H: downstream adaptation is the main recovery locus, but the effect may be mediated by downstream capacity.
T: 8 new seeds, fixed Conv1 full-span replacement, six repair loci, five tau values; exploratory capacity masks and Conv3 severity replication.
D: post-shell locus PASS under preregistered unequal-capacity design; stable post-only recovery UNCERTAIN; pure locus explanation FAIL under equal-capacity follow-up; shell-capacity dependence SUPPORTED.
C: masked-unit optimization may underuse remaining capacity; candidate family is limited to linear Conv1/Conv3; digits/residual-8 specificity remains.
U: n=8 seed uncertainty is moderate; the larger structural uncertainty is the relation between trainable parameter count and functional degrees of freedom.
