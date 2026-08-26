# FAQ

## Is Canaria just pruning or layer dropping?

No. The central empirical question is whether a **composed learned input-output function** can admit a smaller task-preserving replacement than component-wise treatment under a declared experimental grammar. Layer reduction is one possible implementation outcome.

## Does Canaria prove that function composition reduces mathematical complexity?

No. The evidence is operational and task-conditioned. “Simpler” means lower replacement/description cost under explicit tasks, candidate grammars, fidelity/utility rules, and accounting schemes. No Kolmogorov-complexity claim is made.

## What does “compositional simplification” mean here?

In the tested settings, the minimum passing replacement for a composed span can be smaller than the replacement required when source components are fitted separately. The original residual-CNN program, a fresh SmallViT experiment, and a fresh residual-MLP experiment provide the main evidence.

## What is strongest about the residual-MLP replication?

For fresh seeds `1200–1207`, component-wise and composed conditions had **exactly the same learned replacement-parameter budget at every grid point**. Validation alone selected the minimum passing budget; test evaluation followed selection. Composed selected a lower budget in 8/8 seeds, with geometric budget ratio `0.4823×`.

A 2048-parameter joint-factorized control is descriptive/mechanistic secondary. It is consistent with much of the local gap following the composed span objective, but it is not a confirmatory causal decomposition.

## What is the SmallViT boundary?

The locked selection rule used training-held-out NMSE and validation utility, not test accuracy. Across 8/8 fresh eligible seeds the selected composed replacement was smaller, with mean replacement-parameter ratio `0.5199`.

However, the public runner records test accuracy for every candidate. Test was therefore excluded from the preregistered selection criterion but was **not operationally hidden during result generation**. This is weaker isolation practice than the residual-MLP runner and is disclosed in the public replication document.

## Does high Canary identify what can be simplified?

Not reliably enough to make that the project thesis. Strong simplification was observed frequently in low-Canary regions. Treat Canary as a partial sensor, not a necessary condition or established cause.

## Why not simply train the final small model from the start?

In G7, the small-from-start condition was worse than the progressive condition in that cohort. This is a **secondary G7 observation**. The preregistered G7 primary decision compared progressive consolidation against the early and late one-shot controls, and those primary comparisons passed.

## Is progressive consolidation better merely because it uses two compiler fits?

G17 argues against that explanation under the tested protocol. Back-to-back `4→3→2` fitting without task learning between fits was equivalent to direct `4→2`, whereas staged consolidation with intervening task learning was better in G15.

## Does remaining learning horizon universally determine commit timing?

No. G18 supports a narrower claim: the tested deadline-aware controller improved mean PPL and reduced mean compiler updates relative to the tested static NMSE controller. This does not establish a universal commit policy.

## What happened in Phase 2E?

Phase 2E is **not a valid negative result**. It is `INVALIDATED_IMPLEMENTATION_BUG`.

Repair used raw digit input `Xt` where the replacement was defined on internal activation `ta[0]`. Both happened to have width 64, so shape checks did not catch the semantic error. The bugged `0/8` result is retained as history but must not support inference.

See `phase2/README.md` and `../results/phase2/precision_composition/INVALIDATED_HISTORY.md`.

## Did later correction prove that composition needs fewer QAT repair samples?

No. Phase 2O was `UNCERTAIN`: one-sided exact sign-test `p=0.1662`, and the bootstrap95 mean difference crossed zero. A reliable composed repair-sample advantage is not a public claim.

## Are all later Phase 2 raw artifacts in this repository?

No. Phase 2A–C raw protocol/result files and portable runners are in Git. Later 2D–2O correction status is preserved through the correction registry and a correction archive SHA256, but not all later raw per-seed artifacts are checked into this branch. Do not describe those later phases as fully public portable reproductions.

## Can the repository reproduce a confirmatory result without private files?

Yes. G7 seed 4300 has a self-contained runner under `scripts/reproduce/g7_confirmatory/`. In the recorded environment its output exactly matched the archived JSON SHA256 `68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`.

That is software/portability reproduction of an already-confirmatory seed, **not a new independent scientific replication**.

## Has Canaria demonstrated runtime or storage savings?

Only at a bounded small CPU PoC scope. For G7 seed 4300, serialized artifact+manifest size changed `110,093→54,646 B`, and mean batch-128 CPU inference changed `47.05→23.11 ms` across five fresh-process probes.

Meaningful host-RAM reduction was not demonstrated, and GPU/VRAM/energy/LLM/general runtime benefits remain open.

## Is the current branch publication-ready?

Not merely because the experiments exist. The 2026-08-26 independent re-review is a required quality gate. The review must finish with the public-runner smoke test, repository audit, final public-surface update, and closure of Issue #9. PR #7 and the v0.2.0 tag/release boundary are separate repository-state gates.

See `INDEPENDENT_REREVIEW_2026-08-26.md` and `RELEASE_CHECKLIST.md`.

## What should a contributor work on now?

Do not open a broad new experiment family to “complete” the current review. During the quality gate, only fix claim scope, provenance, reproducibility defects, and minimal runner/audit failures. New architecture, scale, theory, and systems questions should begin as explicitly separate follow-up work.

## Safest one-sentence summary

> Canaria reports an operational phenomenon in which some learned spans, under explicit tasks and replacement rules, admit smaller task-preserving replacements when fitted as composed functions than when simplified component-wise; the repository also studies bounded training-time consolidation and a small CPU execution PoC, without claiming universal mathematical complexity reduction or large-model systems benefits.
