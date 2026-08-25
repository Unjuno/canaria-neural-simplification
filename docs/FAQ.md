# FAQ

## Is Canaria just pruning or layer dropping?

No. Canaria includes compression-like interventions, but the central empirical question is whether a **composed learned input-output function** can admit a smaller task-preserving representation than component-wise treatment suggests. Reducing layer count is one possible implementation outcome, not the discovery itself.

## Is Canaria the same as layer fusion?

Not exactly. Layer fusion is an important neighboring idea. Canaria's central claim is about the **measured simplifiability of composed learned computation**, including cases where implementation boundaries are poor functional boundaries and where wider-span replacement succeeds despite weaker local replacement.

## Does high Canary identify what can be simplified?

Not reliably enough to make that the project thesis. Strong simplification occurred frequently in low-Canary regions. The current interpretation treats Canary as a partial sensor of boundary stress/sensitivity rather than a necessary condition or causal mechanism.

## Does the project prove that function composition reduces mathematical complexity?

No. The evidence is operational and task-conditioned. Complexity depends on the task/data distribution, candidate replacement grammar, fidelity/utility criterion, and accounting scheme.

## What does "compositional subadditivity" mean here?

Under a declared replacement/description grammar, the composed span can require less replacement complexity than component-wise treatment would suggest. The original confirmatory setting observed this frequently.

## Why not simply train the final small model from the start?

In the G7 testbed, training the final small architecture from the start was worse than allowing a larger computation to form and consolidating it during learning. This supports separating **learning-time capacity** from **final description capacity** in that setting.

## Is progressive consolidation better merely because it uses two compiler fits?

The G17 equivalence control argues against that explanation. Back-to-back `4→3→2` fitting without task learning between fits was equivalent to direct `4→2`, while G15 staged consolidation with intervening task learning was better.

## What is recontracting?

Continued task learning after a structural consolidation. It reorganizes the remaining system around the new mechanism. Later experiments show a two-sided effect: the next compiler can become easier to optimize while residual errors become more task-sensitive.

## If the next compiler is easier to fit, is the model automatically more robust to replacement error?

No. G20e/G22 show the opposite can occur: matched normalized internal error can cause more immediate task damage after recontracting because downstream sensitivity increases.

## Does teacher-forced perplexity prove an autoregressive replacement is faithful?

No. v22–v25 showed that teacher-forced likelihood can remain close while free-running rollouts diverge substantially. Autoregressive evaluation needs trajectory-sensitive checks when teacher fidelity is the claim.

## Does Canaria work on large pretrained LLMs?

Not established. The project contains small Transformer/decoder/generalization evidence, but the strongest training-time mechanism experiments use a small real-text character-LM testbed. Large pretrained LLM validity remains open.

## Has Canaria demonstrated real runtime speed or VRAM savings?

Not yet as a general systems result. The repository contains real serialized-size results in earlier phases, but runtime compilation/JIT materialization, peak memory, latency, bandwidth, FLOPs, and energy must be measured separately. See `APPLICATIONS.md` and closure Issue #3.

## Why call the replacement procedure a compiler?

Because it translates a learned computation into another functional representation under a declared grammar and budget. It is broader than a conventional source-code compiler and should not be confused with a claim that every neural module can be compiled losslessly.

## Is the 9,926-byte result the entire network?

Yes, for the specific residual-CNN endpoint and declared exact codec, 9,926 bytes is a real round-tripped whole-network serialization. Historical 44.5-byte / ~28-byte figures refer to the compiled core only, not the complete network.

## Is 9,926 bytes a universal minimum?

No. It is an achieved serialized endpoint under one task/architecture/codec, not a codec-independent minimum description length.

## Is there an autonomous Canaria controller?

There are confirmed small-model autonomous-controller experiments. G11 reached the target architecture under a locked non-inferiority protocol, and G18 improved commit timing using remaining learning horizon. Later G21/G27 controls also show that naive safety gates or one fixed risk cap do not automatically solve the cost/utility trade-off.

## Why preserve failed experiments?

Because many of the strongest current conclusions depend on them. Examples include:

- Canary is not necessary;
- unlimited recursive collapse is not supported;
- PPL alone is insufficient for rollout fidelity;
- factorized fitting alone does not explain staged gains;
- matched internal error is not matched task safety;
- a hard damage veto can block contraction.

Removing these failures would make the theory look stronger but less credible.

## What should a new contributor work on?

Start with the three bounded closure directions rather than inventing a new G-number series:

1. clean-repository reproduction (Issue #1);
2. direct replication of compositional simplification on a different family, only if needed for the public claim (Issue #2);
3. minimal runtime-compilation proof-of-concept if deployment claims are pursued (Issue #3).

See `ROADMAP.md`, `OPEN_QUESTIONS.md`, and `CONTRIBUTING.md`.

## What is the safest one-sentence summary?

> Canaria identifies and characterizes task-conditioned compositional simplification in learned neural computation, and shows in small-model experiments that consolidation followed by continued learning can change both the ease and the task-risk of subsequent consolidation.
