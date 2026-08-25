# Publication / communication notes

This document is a claim hierarchy for papers, technical reports, talks, README summaries, and third-party handoff. It is not a manuscript.

## One-sentence project statement

> Canaria identifies and characterizes **task-conditioned compositional simplification** in learned neural computation: some learned computations that are difficult or expensive to simplify component-wise can admit a substantially smaller task-preserving representation when treated as one composed input-output function.

## Primary discovery claim

The central empirical discovery is **not** "we pruned layers" or "we compressed a model." The more distinctive observation is:

> Implementation-level computational complexity can be non-additive across learned boundaries. In the tested settings, composed spans were simpler to replace than component-wise treatment suggested.

Supporting evidence now includes two complementary lines:

1. **Original residual-CNN confirmatory evidence**
   - composition subadditivity `P(G>0)=0.7107`, 95% CI `0.6128–0.8137` under the declared grammar;
   - strong simplification outside high-Canary regions;
   - implementation-block boundaries failing as universal functional boundaries;
   - wider-span replacement succeeding where local replacement could fail;
   - whole-network accounting rejecting the strongest hidden-relocation explanation.

2. **Direct Small Vision Transformer replication**
   - fixed central two-block span;
   - common replacement grammar for component-wise and composed treatment;
   - locked passing criterion: held-out span NMSE `<=0.12`, validation utility `>=0.95`;
   - first 8 fresh baseline-eligible seeds `>=9000`;
   - component-wise minimum passing complexity: **9,808 replacement params** in all 8 seeds;
   - composed minimum passing complexity: **4,904–5,424 params**;
   - mean composed/component-wise ratio: **0.51988**;
   - paired seed-bootstrap95: **[0.50634, 0.53926]**;
   - composed smaller: **8/8 seeds**;
   - selected composed mean test utility: **0.97856**, bootstrap95 **[0.97090, 0.98562]**;
   - component-wise fitting received twice the compiler updates, so the complexity advantage was not purchased by extra optimization on the composed side.

### Safe wording

- "We identify and systematically characterize an empirical phenomenon of compositional simplification in learned neural computation."
- "Under declared task distributions and replacement grammars, composed learned functions can exhibit subadditive replacement/description complexity."
- "The core component-wise-versus-composed effect was directly replicated in a Small Vision Transformer under a fresh locked protocol."

### Avoid

- "Function composition always reduces complexity."
- "We measured Kolmogorov complexity."
- "Canary is the cause or necessary condition of simplification."
- "Every neural network can be compiled to a tiny program."
- "The SmallViT result proves universal Transformer or LLM subadditivity."

## Secondary discovery: dynamic consolidation

The training-time program extends the static observation:

> **form → transfer → commit → recontract → transfer again**

Key evidence:

- G7: progressive consolidation outperformed one-shot schedules and training the final small architecture from the start.
- G15: staged `4→3→2` with task learning between commits beat waiting for direct `4→2`.
- G17: the same two-step compiler factorization without intermediate task learning was equivalent to direct contraction.
- G19: the staged-path effect replicated on `5→4→2` versus direct `5→2` with equal compiler-update budgets.

A defensible interpretation is:

> Structural consolidation changes the subsequent learning trajectory; intervening task learning/recontracting is part of the effect rather than merely post-hoc damage repair.

## Mechanistic refinement

G20–G26 show that "becomes simpler" needs two axes.

### Compiler side

After intermediate task learning, the next compiler can reach the same normalized functional-error target with fewer optimization updates.

### Task side

The same normalized residual error can become **more** damaging because downstream task sensitivity increases.

The strongest current synthesis is:

> Recontracting can make the representation more compiler-friendly while making the task computation sharper with respect to residual approximation error.

Sensitivity-aware quantities outperform error magnitude alone for immediate task-damage prediction, and remaining learning horizon improves future-damage prediction.

Do not present the empirical first/second-order predictor as a proved Taylor law or universal formula.

## Important negative results to mention

A credible public account should include at least these boundaries:

- Canary is not a necessary local condition.
- Unlimited recursive collapse is not supported.
- Teacher-forced PPL can remain nearly unchanged while autoregressive rollouts diverge.
- v23–v25 tested natural-text post-hoc objectives did not close that rollout boundary.
- G21 hard task-damage veto could prevent final contraction.
- G27 fixed risk caps did not yield a cost/utility Pareto improvement.
- The current runtime PoC did not demonstrate meaningful host-RAM reduction.
- The direct SmallViT replication remains small-model and grammar-dependent.

## Reproducibility statement

A public self-contained runner reproduces G7 fresh confirmatory seed 4300 without private `/mnt/data` imports.

In the recorded environment, the complete reproduced JSON exactly matches the archived confirmatory output with SHA256:

`68265c044f51338f616fc6b43380cf0edb44ea142e10f80c66dea5394ded0028`

This is a **reproduction/portability result**, not a new independent scientific replication.

## Application framing

The systems idea is to move from:

> model as stored parameter tensors

 toward:

> model as a compact task-conditioned functional representation that can be serialized, materialized, or compiled for execution.

### What the current PoC supports

A bounded CPU-only PoC on G7 seed 4300 demonstrated:

- serialized artifact + manifest: **110,093 → 54,646 bytes** (`−50.36%`);
- parameters: **23,138 → 11,042** (`−52.28%`);
- mean batch-128 CPU inference: **47.05 → 23.11 ms**;
- direct execution of the compact learned 2-block representation without reconstructing the original 4-block model.

### What it does not support

- meaningful host-RAM reduction: RSS delta was only **4.72 → 4.56 MB**;
- a general cold-start claim: load/materialization was lower on average but cache-sensitive;
- GPU/LLM/energy/VRAM/universal runtime gains.

Publication-safe systems statement:

> In one small CPU proof of concept, a progressively consolidated learned representation was about 50% smaller as a serialized artifact and about 2× faster for the measured batch-128 inference workload, while meaningful host-RAM reduction was not demonstrated.

## Suggested paper / technical-report structure

1. **Problem:** implementation boundaries may overstate task-effective functional complexity.
2. **Discovery:** original compositional simplification / subadditivity.
3. **Direct replication:** SmallViT component-wise versus composed replacement under a locked fresh protocol.
4. **Controls:** Canary not necessary; boundary expansion; whole-network accounting; negative recursive results.
5. **Dynamic extension:** training-time consolidation and staged-vs-direct mechanism separation.
6. **Recontracting mechanism:** compiler conditioning versus downstream sensitivity.
7. **Boundaries:** autoregressive rollout failures and controller failures.
8. **Reproducibility:** public exact reproduction of one confirmatory seed.
9. **Systems implication:** bounded serialization/materialization/direct-execution PoC with explicit RAM boundary.
10. **Limitations / handoff:** small models, operational complexity grammar, large-LLM external validity.

## Novelty / priority language

The repository supports claiming that Canaria **identifies and characterizes** this empirical phenomenon and now includes a fresh cross-family direct replication in a Small Vision Transformer.

A strict "first ever" priority claim should still be made only after a dedicated literature review covering at least layer fusion, depth pruning, progressive module replacement, distillation, network morphism/surgery, learned program extraction, and function-preserving compression.

The novelty should be attached to the phenomenon and the controlled characterization—not to the generic act of reducing layer count.

## Recommended citation practice

Until a paper exists, cite:

- the exact repository commit or public-snapshot tag;
- the relevant protocol/result files or recorded SHA256 identifiers;
- the evidence class (confirmatory/exploratory/negative/reproduction/systems-PoC) when discussing a specific endpoint.
