# Training-time functional consolidation — corrected Canaria mainline

Date: 2026-08-24

## 1. Research correction

The current mainline studies **training-time functional consolidation**, not merely post-hoc deletion/pruning.

The research question is:

> Can a model first use a larger computational core to form useful computation, then transfer a learned span into a smaller replacement, commit the replacement, continue task learning so the surrounding computation recontracts around the new mechanism, and repeat this process while preserving or improving task utility?

This correction followed an audit of the experimental trajectory. Local attention/head deletion and frozen single-component replacement are retained as side diagnostics, but they are no longer treated as the core Canaria question. Earlier evidence had already suggested that implementation boundaries can be poor functional boundaries and that wider-span consolidation can succeed where local replacement fails.

The working algorithmic rule is:

> **form → transfer → commit → recontract → transfer again**

rather than:

> train fully → compress once

or:

> split one compiler fit into several fits without intervening task learning.

## 2. Shared G7–G17 testbed

The current confirmed sequence uses the small real-text character-LM setup inherited from v23–v25.

Architectures:

- Large: 4 blocks, MLP width 48, 23,138 parameters.
- Intermediate: 3 blocks, MLP width 36, 16,502 parameters.
- Final: 2 blocks, MLP width 24, 11,042 parameters.
- Final parameter reduction relative to Large: **52.2776%**.

Task training:

- 12 epochs / 48 task optimizer updates in the controlled G7 family.
- Epoch-specific minibatch order shared within a seed at branch points.
- AdamW state is preserved for unchanged shell parameters across consolidation where the protocol specifies it.
- Replacement cores are fit to span input/output activations using training/calibration data only.
- Held-out test data is not used to choose autonomous commit times.

The compiler-cost accounting used in these experiments is primarily a **replacement-parameter-count × optimizer-update proxy**. It is not exact FLOPs, energy, or wall-clock equivalence.

## 3. G7 — training-time consolidation

### Question

Does a network benefit from first learning with a larger computational core and then consolidating computation during training, compared with training the final small model from the start or using a one-shot consolidation?

### Confirmatory design

Fresh seeds: 4300–4307.

Conditions:

1. `large_reference`: 4×48 for all 12 epochs.
2. `small_from_start`: 2×24 for all 12 epochs.
3. `terminal_posthoc`: 4×48 for 12 epochs, then 4→2 fit, no task recovery.
4. `late_one_shot`: 4×48 through epoch 8, then 4→2, task epochs 9–12.
5. `early_one_shot`: 4×48 through epoch 4, then 4→2, task epochs 5–12.
6. `progressive_compute_matched`: epoch 4: 4→3; task epochs 5–8; epoch 8: 3→2; task epochs 9–12.

Direct 2-block replacement used 40 fit epochs. Progressive used 15 fit epochs for 3×36 and 14 for 2×24. The replacement-parameter × optimizer-update proxy differed by about +0.20%.

Primary PASS rule: paired bootstrap 95% CI for progressive minus early one-shot and progressive minus late one-shot must both lie entirely below zero.

### Results

Mean test PPL:

| condition | mean PPL |
|---|---:|
| large reference | 19.7843 |
| small from start | 21.3126 |
| terminal post-hoc | 19.7978 |
| late one-shot | 19.8675 |
| early one-shot | 19.7515 |
| progressive | **19.4478** |

Paired differences:

- progressive − early: **−0.3037**, 95% CI **[−0.3384, −0.2664]**, 8/8 lower.
- progressive − late: **−0.4197**, 95% CI **[−0.5114, −0.3229]**, 8/8 lower.
- progressive − large (secondary): **−0.3365**, 95% CI **[−0.4175, −0.2492]**, 8/8 lower.
- terminal post-hoc − large: **+0.0134**, 95% CI **[−0.0021, +0.0291]**.
- small-from-start − large: **+1.5282**, 95% CI **[+1.0855, +1.8885]**.

Decision: **PASS**.

### Recontracting signature

Progressive mean test PPL:

- immediately after first 4→3 consolidation: **26.6813**
- after task recovery through epoch 8: **20.5601**
- immediately after second 3→2 consolidation: **20.6226**
- final after task recovery: **19.4478**

The first consolidation initially damages utility strongly, but continued task learning recovers it and the final smaller model outperforms the tested one-shot schedules.

## 4. G8 — function-aligned transfer is required

### Question

Is the G7 effect merely an architecture curriculum, or must the smaller replacement receive the correct learned function?

Fresh seeds: 4500–4507.

The 4→3→2 architecture schedule and compiler-fitting compute were held fixed. Only the fit target changed:

- `functional`: true span output.
- `identity`: span input.
- `shuffled`: true span outputs permuted across calibration examples.

Mean final PPL:

| condition | PPL |
|---|---:|
| functional | **19.7432** |
| identity | 29.7099 |
| shuffled | 22.7611 |
| untouched large reference | 20.0650 |

Paired results:

- functional − identity: **−9.9667**, 95% CI **[−10.5572, −9.3989]**, 8/8 lower.
- functional − shuffled: **−3.0179**, 95% CI **[−3.3287, −2.7020]**, 8/8 lower.

Decision: **PASS**.

Interpretation: the architecture schedule alone is insufficient under these controls. Some form of function-aligned transfer is materially important. This does not prove that MSE distillation is uniquely necessary.

## 5. G9 — how accurate must the handoff be?

The G7 progressive fit is treated as the 100% compiler-fitting reference budget. Percentages refer to compiler-fitting effort only, not total training compute.

Fresh seeds: 4700–4707.

| budget | mean PPL | stage-1 terminal MSE | stage-2 terminal MSE |
|---|---:|---:|---:|
| 25% | 20.3412 | 0.08787 | 0.12781 |
| 50% | 19.8401 | 0.03865 | 0.05298 |
| 100% | 19.6037 | 0.02091 | 0.02484 |
| 200% | **19.5071** | **0.01143** | **0.01262** |

Paired final-PPL improvements:

- 25→50%: **0.5011**, 95% CI **[0.3975, 0.6690]**, 8/8.
- 50→100%: **0.2364**, 95% CI **[0.2015, 0.2821]**, 8/8.
- 100→200%: **0.0966**, 95% CI **[0.0776, 0.1186]**, 8/8.

Decision: **PASS**.

The dose response is clear, but the marginal return falls strongly. Perfect zero-error transfer is not required before task learning resumes.

## 6. G10 — structured weight inheritance

### Question

Can trained weights be inherited structurally instead of explicitly fitting the span function?

Fresh seeds: 4900–4907.

Conditions:

- `functional`: random compiler initialization + correct span input/output fit.
- `inherit`: deterministic block inheritance, exact attention/LayerNorm copy, narrower MLP keeps high-contribution neurons; no calibration fit.
- `hybrid`: same structured inheritance followed by the same correct functional fit budget as `functional`.

Mean PPL:

| condition | mean PPL |
|---|---:|
| functional | 19.4234 |
| inheritance only | 20.6460 |
| hybrid | **19.3018** |

Paired results:

- inherit − functional: **+1.2226**, 95% CI **[+1.0883, +1.3833]**.
- hybrid − functional: **−0.1216**, 95% CI **[−0.1831, −0.0691]**, hybrid wins 8/8.

Decision: **PASS** for both preregistered hypotheses.

Interpretation: inheritance alone does not preserve enough function, but inheritance gives a better starting point for a short function-aligned refinement.

## 7. G11 — autonomous controller

### Policy

The stable controller uses a calibration-only functional-error threshold and does not inspect the test set. If a candidate does not pass, it is retained and can continue to track the evolving teacher at later checks instead of being reinitialized from scratch.

Fresh seeds: 5400–5407.

Locked criteria:

1. all 8 runs reach the final 2-block architecture;
2. Auto/Large relative PPL has 95% CI upper bound below +2%;
3. compiler updates ≤232 per seed.

Results:

- final 2-block reached: **8/8**.
- mean Auto PPL: **20.1767**.
- mean Large PPL: **20.2267**.
- mean relative Auto/Large difference: **−0.240%**.
- 95% paired bootstrap CI: **[−0.593%, +0.153%]**.
- mean compiler updates: **180**; maximum: **192**.

Decision: **PASS**.

Commit epochs varied across seeds, showing that this was not merely a fixed epoch schedule.

## 8. G15 — staged consolidation versus waiting for one direct merge

Fresh seeds: 5800–5807.

Comparison:

- Staged: `4→3`, task learning, then `3→2`.
- Direct-wait: remain at 4 blocks while a 4→2 candidate is tracked until functional-error thresholds permit commit.

Both finish at the same final architecture.

Results:

- mean staged PPL: **19.9029**.
- mean direct PPL: **20.2022**.
- staged − direct: **−0.2993 PPL**.
- 95% CI: **[−0.3379, −0.2616]**.
- staged wins: **8/8**.
- mean compiler updates: staged **186**, direct **192**.

Decision: **PASS**.

Waiting until a larger one-shot merge becomes accurate enough is inferior under this protocol.

## 9. G17 — is staged fitting alone enough?

G15 still allowed the hypothesis that two smaller compiler fits are simply easier than one large fit. G17 removes task learning between the two consolidations.

Fresh seeds: 6000–6007.

Both conditions:

- commit the final 2-block model at epoch 4;
- use exactly **192 compiler updates**;
- have the same remaining task-training horizon.

Conditions:

- Direct: `4→2`, 12 compiler-fit epochs.
- Factorized: `4→3` for 6 fit epochs, immediately `3→2` for 6 fit epochs, **no task learning between compilations**.

Preregistered equivalence band: **[−0.10, +0.10] PPL**.

Results:

- mean factorized − direct difference: **+0.0279 PPL**.
- 95% CI: **[−0.0019, +0.0614]**.
- full CI lies inside the equivalence band.

Decision: **PASS (equivalence)**.

## 10. Current mechanistic interpretation

The combined evidence supports three parts:

1. **Functional handoff matters.** Changing architecture alone or copying weights alone is insufficient under the tested controls.
2. **A replacement does not need to be perfect before commit.** Continued task learning can repair substantial consolidation error.
3. **The learning interval between consolidations matters.** G15 shows a staged advantage; G17 shows that merely factorizing the compiler fit without intervening task learning does not reproduce it.

The most conservative interpretation is that the network can **recontract / reorganize around a committed smaller mechanism**, and this changed learning path improves the next consolidation and/or final generalization.

The internal cause remains unresolved. Candidate explanations include:

- architecture regularization;
- optimization-basin migration;
- implicit capacity curriculum;
- representation redistribution;
- redundancy re-aggregation around the new boundary;
- genuine reduction in task-conditioned functional description length.

The last two are closest to the strongest Canaria-specific interpretation and require direct tests.

## 11. Side diagnostics kept outside the mainline

R26–R29 local TinyStories replacement experiments are retained as side studies. They showed that frozen local FIR replacement and simple head pruning can preserve local teacher-path quantities much better than free-running trajectory fidelity. These experiments remain useful for understanding autoregressive sensitivity, but they should not be treated as the main evidence for training-time Canaria.

## 12. Evidence boundaries

Do not generalize the G7–G17 result beyond its current scope:

- small character-level real-text LM;
- one principal dataset family;
- source architecture mainly 4 blocks;
- compiler cost measured by update/parameter proxy rather than exact system cost;
- no large pretrained Transformer confirmation yet.

A positive result on this testbed is evidence for a mechanism worth generalizing, not a universal theorem.

## 13. Current next step

The stable G11 policy commits based on current functional NMSE. G13–G17 show that commit quality depends on **remaining task-learning/recontracting horizon** as well as instantaneous fit quality.

The next high-value test is therefore a recontracting-aware policy that estimates expected recovery after commit and chooses among candidate consolidations without test-set access.

See `docs/NEXT_EXPERIMENTS_AUTONOMOUS.md`.
