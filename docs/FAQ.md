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

However, the runner records test accuracy for every candidate. Test was excluded from the preregistered selection criterion but was **not operationally hidden during result generation**. This is weaker isolation practice than the residual-MLP runner and is disclosed in the replication document.

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

No. Phase 2O was `UNCERTAIN`: one-sided exact sign-test `p=0.1662`, and the bootstrap95 mean difference crossed zero. A reliable composed repair-sample advantage is not a current claim.

## Are all later Phase 2 raw artifacts in this repository?

No. Phase 2A–C raw protocol/result files and portable runners are in Git. Later 2D–2O correction status is preserved through the correction registry and a correction archive SHA256, but not all later raw per-seed artifacts are checked into the current branch.

## Can the repository reproduce a confirmatory result without private files?

Yes for the existing G7 portability path. G7 seed 4300 has a self-contained runner under `scripts/reproduce/g7_confirmatory/`; in its recorded environment its output exactly matched the archived JSON SHA256 `68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`.

That is reproduction of an already-confirmatory seed, **not a new independent scientific replication**.

For the residual-MLP headline cohort, a pinned full-cohort reproduction gate is being hardened separately; a one-seed smoke test is not enough for announcement readiness.

## Has Canaria demonstrated runtime or storage savings?

Only at a bounded small CPU PoC scope. For G7 seed 4300, serialized artifact+manifest size changed `110,093→54,646 B`, and mean batch-128 CPU inference changed `47.05→23.11 ms` across five fresh-process probes.

Meaningful host-RAM reduction was not demonstrated, and GPU/VRAM/energy/LLM/general runtime benefits remain open.

## Is the repository announcement-ready?

No. The current integrated gate is [`ANNOUNCEMENT_READINESS.md`](ANNOUNCEMENT_READINESS.md) / Issue #13. The historical `v0.2.0-public-snapshot` tag and archived release checklist document an earlier repository boundary; they are not current approval to announce.

The 2026-08-26 independent re-review remains the baseline claim audit, but current hardening still includes full-cohort pinned reproduction, a decision on weak external-validity evidence, repository-surface cleanup, and a final integrated review.

## Where did the old v10–v25 directories go?

They are preserved under `../archives/research-history/`. They were moved out of active navigation during pre-announcement cleanup so a new reader does not confuse version-number archaeology with the current evidence surface. See `../archives/README.md`.

## What should a contributor work on now?

Prefer a specific blocker or falsifiable scientific weakness over broad experiment accumulation. Current work should either strengthen a known weak evidence boundary, fix reproducibility/provenance, or simplify the public surface. New research should remain isolated until reviewed.

## Safest one-sentence summary

> Canaria reports an operational phenomenon in which some learned spans, under explicit tasks and replacement rules, admit smaller task-preserving replacements when fitted as composed functions than when simplified component-wise; the repository also studies bounded training-time consolidation and a small CPU execution PoC, without claiming universal mathematical complexity reduction or large-model systems benefits.
