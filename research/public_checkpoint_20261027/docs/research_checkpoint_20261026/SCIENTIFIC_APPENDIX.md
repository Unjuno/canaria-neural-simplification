> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Scientific appendix: claim–evidence matrix

**Scope:** finite consolidation of already-existing text records, not a new experiment, model reproduction, audit replay, or independent confirmation. All numerical observations below are attributed to their pinned records. The manifest records local Git identities; it does not claim remote publication or private-binary availability.

## 1. Claim–evidence matrix

IDs identify groups in `EVIDENCE_MANIFEST.json`. Exact full commit/path/blob/SHA-256 identities are supplied there; abbreviated commits below are reading aids only.

## 2. Decisive F/D evidence, without causal overclaim

F is endpoint MSE; D is the pure probability-weighted, one-sided teacher-gap objective. Counts are from the same 1,000 diagnostic IDs for each teacher, not three independent image samples.

| Seed | Teacher correct | F → D correct | F → D harms | Q1 harms F → D | Q2–Q5 harms F → D | Q5 harms F → D |
|---|---:|---:|---:|---:|---:|---:|
| 26102601 | 920 | 884 → 806 | 55 → 132 | 35 → 56 | 20 → 76 | 0 → 2 |
| 26102602 | 918 | 889 → 824 | 53 → 119 | 41 → 56 | 12 → 63 | 0 → 3 |
| 26102603 | 910 | 875 → 700 | 59 → 241 | 38 → 67 | 21 → 174 | 0 → 13 |

All three fail ≥20% Q1 harm reduction, Q2–Q5 nonincrease, Q5 nonincrease and +10 correct/1,000.

The independent auditor cross-evaluated D on saved diagnostic logits, using the locked fit μ. This is an already-completed descriptive calculation, not a calculation repeated for this checkpoint:

| Seed | Full-fit D, F → D (reported) | Diagnostic D, F → D (auditor-derived) | Diagnostic centered logit RMS, F → D |
|---|---:|---:|---:|
| 26102601 | 5.091620 → 1.133534 | 4.288933 → 6.335783 | 4.716069 → 3.637913 |
| 26102602 | 4.368271 → 0.771342 | 5.348495 → 9.136676 | 4.762206 → 3.928450 |
| 26102603 | 4.890994 → 2.546136 | 4.411141 → 4.384239 | 4.788992 → 2.085312 |

This is **not** uniformly “better diagnostic surrogate, worse accuracy”: diagnostic D worsens in seeds01/02, while seed03 has only a slightly better aggregate diagnostic D despite substantially lower accuracy. Every Q2–Q5 bin has worse diagnostic D under D. Measured logit scale contracts, so these endpoints do not support a logit-explosion explanation.

Objective geometry, probability weighting, scale/conditioning, capacity and optimization remain joint explanations, not individually identified causes. Declining late fit loss does not prove convergence or that longer training would restore diagnostic accuracy. Positive nonzero surrogate loss is not a decision-preservation guarantee. The negative result is not dismissed as an arithmetic artifact, but the static audit does not exhaustively exclude every subtle training fault.

## 3. Evidence hierarchy and unresolved debt

1. **Original confirmation versus technical replay:** original R87R2/Phase4 confirmatory labels describe the original locked cohorts. Their subsequent exact replays provide technical reproducibility, not additional independent scientific n. Older archive recovery remains separate.
2. **Integrity versus science:** arithmetic/hash/source/process-custody checks establish consistency within their stated scopes, not external peer review, signed historical attestation, absence of unlogged runs, or independent model execution.
3. **Decision audit limits:** the accepted audit used saved logits, logs and metadata only. Raw fit targets, μ tensors and models were not deserialized. Full-fit D is a reported field, not independently regenerated. Endpoint SSE/teacher sums were checked arithmetically, not independently reproduced from endpoint tensors. Filesystem times corroborate control-flow/hash ordering; they are not signed proof.
4. **Quality versus size:** a smaller replacement is not automatically a smaller useful deployed model. Native-cost evidence is for R87R2 artifacts, not CIFAR. No successful CIFAR utility-qualified whole-model deployment claim follows.

## 4. Research decision and a legitimate reopening condition

The current CIFAR intervention branch is scientifically complete through its negative results. Further seen-data width, budget, temperature, margin-weight or loss-mixture tuning is not the default continuation. This closes this intervention branch, **not the entire three-day research effort** and not the general idea.

A new fitting proposal must specify one distinct mechanism-based hypothesis, explain what the present evidence can and cannot discriminate, freeze comparisons/failure census/resource budget, and provide a credible unexposed confirmation route. Original validation is seen, calibration diagnostics are seen, and official test is unblinded; repartitioning those observations does not create fresh confirmation. No new untouched source is established by this consolidation. For a deployment claim, acceptable task quality and relevant whole-model resource evidence must concern the same frozen artifacts.

## 5. Supersession and preservation notes