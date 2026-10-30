# Fresh-source reproduction status — 2026-10-29

**Status: preparation failed; no new scientific result.**

A one-shot technical reproduction was admitted against executable commit
`06301e653b69151107af39100e0d4a82445867c5`, with a single 30-minute deadline
and at most two scientific workers. This was a reproduction on previously seen
Digits, not independent-data confirmation.

## What was tested

The new outer supervisor passed five bounded benign-process checks: normal
completion, refusal to reuse a claimed attempt, capture of an orphaned child in a
separate session, TERM-to-KILL cleanup, and deadline enforcement while waiting
for review. These checks used no scientific data or models. An earlier initial
probe failed during initialization before child launch; its original stderr was
not retained. Subsequent inspection found the assumed process-children
interface unavailable. That failure and diagnostic gap were retained. The
revised supervisor used read-only process ancestry metadata and identity-bound
process handles.

## What failed, and why

The admitted pipeline failed during preparation of a separate clean checkout.
The existing repository was shallow, so its local Git clone operation created
additional helper subprocesses. Six owned processes were observed against the
preparation stage's frozen three-process cap. The supervisor terminated the
owned process tree and recorded failure.

The preparation design was incorrect: its total-process budget did not account
for Git's helper topology. A cap on scientific workers is not a cap on all
coordinator and repository-preparation processes. This was an orchestration
failure, not evidence that the scientific method failed to reproduce.

All six observed processes were reaped; the terminal record contained no
remaining owned processes and was not late. This is scoped observed cleanup,
not a guarantee about unrelated host activity or all possible OS failure modes.
The original executable checkout remained clean and unchanged. Partial evidence
and the consumed attempt were retained; there was no automatic retry or cap
relaxation.

## What remains unverified

The attempt did not complete runtime/resource preflight, train teachers, run
matched comparisons or the budget grid, create selected-model storage results,
or measure latency. Consequently, it does not establish fresh end-to-end
reproduction and changes none of the previously reported scientific estimates.

A future separately reviewed attempt would need a preparation method or helper
budget that fits the actual repository-copy topology, while preserving the
scientific worker limit, immutable source identity, finite deadline and
independent stage reviews. This is a proposed repair, not an executed result or
permission to reuse the consumed attempt.
