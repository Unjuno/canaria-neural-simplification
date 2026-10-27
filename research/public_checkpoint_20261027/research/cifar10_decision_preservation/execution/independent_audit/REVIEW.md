> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Independent saved-output audit: fixed decision preservation

**PASS_WITH_STATED_LIMITS — 528,776 saved-evidence checks; no mismatch found.** This is an audit of recorded arithmetic and custody, not an independent model reproduction.

The prespecified fixed prediction failed in all three complete teacher pairs. Every substantive criterion failed per seed: low-margin harm reduction, Q2–Q5 nonincrease, Q5 nonincrease, and +10 correct/1,000. This does not authorize a new fit or changed threshold.

## Independently recomputed outcomes

F = feature MSE; D = decision-preservation. The same 1,000 previously seen diagnostic IDs were used for each teacher.

| Seed | Teacher correct | F → D correct | F → D harms | F → D gains | F → D Q1 harms | F → D Q2–Q5 harms | F → D Q5 harms |
|---|---:|---:|---:|---:|---:|---:|---:|
| 26102601 | 920 | 884 → 806 | 55 → 132 | 19 → 18 | 35 → 56 | 20 → 76 | 0 → 2 |
| 26102602 | 918 | 889 → 824 | 53 → 119 | 24 → 25 | 41 → 56 | 12 → 63 | 0 → 3 |
| 26102603 | 910 | 875 → 700 | 59 → 241 | 24 → 31 | 38 → 67 | 21 → 174 | 0 → 13 |

Accuracy differences D−F are −7.8, −6.5 and −17.5 percentage points. Lost F corrections are 9/11/13; new D-only corrections are 8/12/20 (shared corrections 10/13/11), so gains do not offset the added harms.

## Descriptive objective/scale findings — no new success criterion

Diagnostic D below was independently calculated from the saved diagnostic teacher/student logits using the exact all-nine-rival one-sided squared erosion formula and each frozen fit μ. It is not a new fit or inference. Full-fit D is the saved reported value; fit student logits are not available in JSON.

| Seed | Fit D, F → D (reported) | Diagnostic D, F → D (recomputed) | Diagnostic NMSE, F → D | Student logit RMS, F → D |
|---|---:|---:|---:|---:|
| 26102601 | 5.091620 → 1.133534 | 4.288933 → 6.335783 | 0.277413 → 0.627408 | 4.716069 → 3.637913 |
| 26102602 | 4.368271 → 0.771342 | 5.348495 → 9.136676 | 0.231632 → 0.670952 | 4.762206 → 3.928450 |
| 26102603 | 4.890994 → 2.546136 | 4.411141 → 4.384239 | 0.253657 → 0.981383 | 4.788992 → 2.085312 |

- Seeds 01 and 02 worsen even the diagnostic surrogate. Seed 03 has a slightly lower aggregate diagnostic surrogate, despite much worse accuracy.
- Seed 03 reduces Q1 diagnostic D (21.016322 → 18.398314) but worsens Q1 harms (38 → 67). Every Q2–Q5 bin has larger D surrogate under the D arm in all three seeds. This illustrates why the weighted loss is not a thresholded accuracy guarantee.
- Observed student logit scale **contracts**, not explodes. No logit-explosion explanation is supported by these saved endpoints.
- All six logs contain 2,048 consecutive updates, 256,000 presentations each, 64 complete passes, and no zero loss or zero gradient updates. D gradients are finite but on a different numerical scale from F; the objectives have different units. No convergence, causal attribution or optimizer-defect conclusion follows.

## Audit coverage and custody

- Independently checked all 6,000 per-arm diagnostic rows and 54,000 rival erosions against raw saved logits; first-index argmax, ties, accuracy, agreement, H/G, overlap and correction pairing match.
- All 15 teacher-only 200-image bins and all 12 boundary records exactly match sorting by teacher top1−top2 margin then image ID; teacher logits match the prefit old saved teacher cache.
- All 42 fit/diagnostic/quintile NMSE records satisfy the sufficient-statistic arithmetic; quintile SSE totals equal overall diagnostic SSE. Full-fit feature MSE matches saved SSE within the declared audit reduction tolerance (float32 mean vs float64 sum).
- Frozen μ values are 0.04994397945218263 / 0.04916923945471031 / 0.051111225107462084. The target raw payload hashes, identity JSON, identical paired initializations, session locks, source implementation and pre-optimizer ordering agree. No target or checkpoint was deserialized.
- Saved supervisor records show serial bounded strata/fit/report children, exit 0, reaped, no retry and no deadline expiry. All six terminals precede reporting.
- The 30-minute report allocation and original cohort/resource deadlines were unchanged. No data/model operation, archive opening, network or CI occurred in this audit.

## Evidence limitations

- Saved outputs only; no inference, model/target/checkpoint deserialization, archive read, new fit, new test, network or CI.
- All 6,000 per-arm diagnostic rows, saved logits and 54,000 rival erosions checked; teacher rows span three views of the SAME 1,000 IDs, not independent new samples.
- NMSE recomputed from 42 saved sufficient-statistic records; original endpoint tensors are not saved in JSON, so SSE/centered teacher SS are not independently regenerated.
- Full-fit mu and target tensors are opaque .pt payloads: raw SHA256 and pre-optimizer identity/lock/source bindings verified, but mu numerical reconstruction requires deserialization and was intentionally not done.
- Full-fit D has no saved fit student logits in JSON, so its numeric value is a report field, not independently regenerated. Diagnostic D is newly derived only from saved diagnostic logits using locked fit mu; it is descriptive and does not replace the prespecified criteria.
- Filesystem ordering corroborates source control-flow and hash bindings; filesystem times are not signed independent timestamps.
- No posthoc threshold, new utility pass, significance, generalization or causal claim. Seen calibration diagnostics; historical original-validation/official-test negatives remain unchanged.
- Resource compliance is verified from saved supervisor/clock records and source identities, not a retrospective independent measurement of every process/thread.

## Artifacts

- `AUDIT.json`: check counts, exclusions, hashes and clock evidence.
- `audit_saved_outputs.py`: self-contained stdlib auditor, no project imports.
- `counts_and_quintiles.json`, `objective_and_scale_table.json`, `fixed_criteria.json`, `shared_unique_harms.json`, `CORRECTION_PAIRING_AUDIT.json`: complete tables.
- `sample_audit_derived.json`: all per-arm rows plus descriptive diagnostic D.
- `training_log_descriptive.json`: every pass summary and terminal diagnostics.
- `INPUT_IDENTITIES.json`, `COMMIT_IDENTITIES.json`: raw input and local commit identities.

Report SHA256: `ea1ad3cb9b0fc6baa0c32e85f3bda15d31e6c82666c13436aa57718dc4f4ea66`.

Audit implementation note: an initial read-only audit invocation encountered the predecessor cache field name `teacher_logits` instead of `logits`; this local audit-parser schema mismatch was corrected. No experiment was retried, source/result artifact changed, or scientific threshold modified.
