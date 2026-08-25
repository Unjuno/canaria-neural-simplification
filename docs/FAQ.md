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

Continued task learning after structural consolidation. It reorganizes the remaining system around the new mechanism. Later experiments show a two-sided effect: the next compiler can become easier to optimize while residual errors become more task-sensitive.

## If the next compiler is easier to fit, is the model automatically more robust to replacement error?

No. G20e/G22 show the opposite can occur: matched normalized internal error can cause more immediate task damage after recontracting because downstream sensitivity increases.

## Does teacher-forced perplexity prove an autoregressive replacement is faithful?

No. v22–v25 showed that teacher-forced likelihood can remain close while free-running rollouts diverge substantially. Autoregressive evaluation needs trajectory-sensitive checks when teacher fidelity is the claim.

## Does Canaria work on large pretrained LLMs?

Not established. The project contains small Transformer/decoder/generalization evidence, but the strongest training-time mechanism experiments use a small real-text character-LM testbed. Large pretrained LLM validity remains open.

## Can the public repository reproduce any confirmatory result without private files?

Yes. G7 fresh confirmatory seed 4300 has a self-contained runner under `scripts/reproduce/g7_confirmatory/`.

In the recorded environment, the complete reproduced JSON exactly matched the archived confirmatory output with SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

This validates portability of an already-confirmatory seed; it is not a new independent scientific replication.

## Has Canaria demonstrated real runtime speed or storage savings?

At a **small CPU-only PoC scope**, yes for some metrics.

For G7 seed 4300:

- serialized artifact + manifest: `110,093 → 54,646 B` (`−50.36%`);
- batch-128 CPU inference: `47.05 → 23.11 ms mean` over five fresh-process probes;
- the compact 2-block learned representation executes directly and does not reconstruct the original 4-block model.

See `RUNTIME_POC.md`.

## Has Canaria demonstrated RAM/VRAM savings?

No meaningful host-RAM reduction was demonstrated in the current PoC. Process RSS delta changed only `4.72 → 4.56 MB` (`0.966×`). GPU VRAM has not been benchmarked.

This is why the repository keeps storage size, runtime latency, and memory as separate claims.

## Is load/materialization definitely faster?

Not as a general claim. The five-process benchmark averaged `7.85 → 5.86 ms`, but auxiliary runs showed cache/filesystem sensitivity. It remains secondary evidence rather than a headline claim.

## Does the runtime PoC prove universal speedup?

No. It is one small model, one seed, CPU-only, and one batch-size workload. It does not establish GPU, LLM, energy, browser, edge, or universal runtime benefits.

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
- a hard damage veto can block contraction;
- the current compact runtime PoC does not demonstrate meaningful host-RAM reduction.

Removing these failures would make the theory look stronger but less credible.

## What should a new contributor work on?

The repository-portability task (Issue #1) and minimal runtime PoC (Issue #3) are complete.

Only one optional closure task remains: **Issue #2**, a direct replication of compositional simplification on a clearly different architecture/task if a stronger publication-level generalization/novelty claim is deliberately pursued.

Otherwise, choose a question from `OPEN_QUESTIONS.md` as a new research project rather than extending an indefinite G-number sequence.

## What is the safest one-sentence summary?

> Canaria identifies and characterizes task-conditioned compositional simplification in learned neural computation, shows in small-model experiments that consolidation followed by continued learning changes both the ease and task-risk of later consolidation, and provides a bounded CPU PoC showing that one compact learned representation can be serialized and executed directly without reconstructing its larger predecessor.
