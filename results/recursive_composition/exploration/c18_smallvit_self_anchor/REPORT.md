# C18 — self-anchored interface transfer to a SmallViT hierarchy

Status: **EXPLORATORY MIXED RESULT — mechanism-positive, utility-failure**.

C18 asked whether the self-anchored compressed-interface repair discovered in the residual-MLP recursive hierarchy transfers qualitatively to a SmallViT-family hierarchy.

## Eligibility

Fresh seeds were `1490–1492` with locked teacher validation eligibility `>= 0.95`.

- 1490: eligible, teacher val accuracy 0.9556.
- 1491: ineligible, teacher val accuracy 0.9444; retained and not replaced.
- 1492: eligible, teacher val accuracy 0.9556.

No held-out test evaluation was performed.

## Functional ordering

Both eligible seeds produced the same final hidden-NMSE ordering:

`full_32 < anchored_16 < anchored_8 < frozen < sketch_only_16`

For `anchored_16`:
- final NMSE improvement vs frozen: −0.01993 and −0.01351;
- final/full-32 ratios: 1.02287x and 1.02138x.

For `anchored_8`:
- final NMSE improvement vs frozen: −0.00990 and −0.00518;
- final/full-32 ratios: 1.03748x and 1.03327x.

The naive unanchored 16D sketch worsened final NMSE vs frozen by +0.04380 and +0.06931, reproducing the qualitative complement-drift failure seen in residual-MLP C11.

## Absolute utility failure

The result cannot support useful full-span SmallViT compression. The tested exact-4096-parameter token-wise grammar is too weak for the entire four-block teacher span:

- direct-original matched final validation accuracy: 0.10 in both eligible seeds;
- full-32 recursively aligned final validation accuracy: 0.10 in both eligible seeds.

Thus the hidden-interface mechanism transferred qualitatively, but the task-level replacement grammar collapsed to chance.

## Execution provenance

The protocol and GitHub runner were locked before outcomes. Local execution used a logic-equivalent standalone mirror because the GitHub connector and execution container do not share a filesystem; the committed runner itself was not executed byte-identically. This is acceptable for exploratory diagnosis but is an additional reason not to use C18 as confirmatory evidence.

## Decision

Do not run a confirmatory version of this full-four-block testbed. The next experiment should preserve the established SmallViT two-block replacement regime from C4/C5, where task utility is materially higher, and test the self-anchored compressed-interface signal there as a relative mechanism control.
