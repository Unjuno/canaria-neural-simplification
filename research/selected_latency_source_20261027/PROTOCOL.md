# Frozen scientific rules; separately versioned operational handoff

## Original scientific procedure (preserved)

Use all 16 already-fixed teachers and their 32 selected C/J stored, uncompressed
whole-model NPZ archives. The deployed teacher has stem, blocks 0–3 and head;
C/J retain blocks 2–3 with the original replacement modules. C uses TinyRes width
budget/128; J uses two TinyRes units each width budget/256. Keys, shapes, float32,
finiteness, source/archive hashes and parameter counts must match. All 48 graphs
must agree within maximum absolute error 1e-5 with their saved 270-validation-row
logits, in chunks 256 and 14, before timing starts. Read only the validation feature
columns; do not consume labels or the original 450-test metrics.

Python 3.13.5, torch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0, SciPy 1.17.0.
CPU eager AVX2, MKL COMPATIBLE, mkldnn disabled, deterministic algorithms,
one intra-op and one inter-op thread, eval mode and no_grad. No compilation,
quantization, kernel tuning, affinity tuning, training or new selection.

Synthetic normalized input is exactly
`np.random.default_rng(20261027).integers(0,17,size=(128,64)).astype(np.float32)/16`.
Batch 1 is the first row; all models/seeds share input. Input generation, loading,
hashing, correctness, serialization and summaries are outside the timed blocks.
Each model/batch has 20 warmup forwards; each of 30 rounds has a block of exactly
20 forwards per arm timed with perf_counter_ns. No adaptive iteration count.

The exact frozen order is `list(itertools.permutations(('teacher','C','J')))`;
choose entry `(seed+round)%6`. Thus all six permutations occur five times for each
seed/batch. The old prose called this lexicographic; the literal arm enumeration,
NOT alphabetically sorted strings, is authoritative. Seeds 0–15 ascend. Even seed
IDs use batches (1,128), odd seed IDs (128,1). Preserve all 96 cells and all 2,880
blocks with round/position/order/elapsed and per-forward values, including outliers.

Per-seed/batch/arm latency is the median of 30 block ns/forward values. Report all
16 paired C/J, C/teacher, J/teacher ratios per batch, plus their geometric means,
min/max and smaller/tie/larger counts. No exclusion, trimming, winsorization,
significance test, result-driven repeat or claim about training time.

## Original lifecycle provenance — not reopened

The original common deadline was 2026-10-29 06:50 UTC. The historical registry is
already consumed; a new path or release does not reset it. Original release and
source/registry checks are not represented as rerun by this derivative.

## Portable lifecycle v1 — future execution requires separate authority

A separately approved release binds a new technical-attempt ID, canonical private
control-root identity, explicit UTC deadline, exact input-manifest SHA256, every
package file hash, and the clean reviewed implementation commit. Approval booleans
are attestations requiring human verification, not signatures or an authority grant.
No supplied example is a valid release. The new deadline is explicit and cannot
be mistaken for an extension of the original scientific approval.

Only supervise.py may launch the worker. Hold nonblocking supervisor and benchmark
flocks in the authorizer-selected shared control root throughout the attempt.
Never rotate that root to evade exclusivity; it must cover every concurrent
benchmark in scope. Atomically create one permanent attempt registry; reject any
existing registry or output; never delete/reset/resume/retry. Output/control/input
and repository trees must not overlap. The release hashes the canonical control
root's UTF-8 path, preventing unchanged-release replay in a new directory.

Preserve the 600s envelope: at most 590s worker window including setup/native calls,
then process-group TERM with 5s wait, KILL with 5s wait; deadline remainder reduces
it. A Python alarm is secondary. Worker address space is capped at 4GiB. Evidence
across output+registry is capped at 128MiB before writes, with 1MiB terminal reserve,
including 64KiB for supervisor rescue. Every failure retains a conservative full
96-cell census including NOT_STARTED, and never promotes a partial cohort.
Malformed/truncated worker metadata triggers conservative supervisor failure
rather than a false completion. Exceptional postlaunch paths terminate/reap the
worker; an unreapable worker or inaccessible evidence storage is an explicit
unrecoverable failure, never permission to start another attempt.

CPU, affinity, load and process census are captured before/after in PRIVATE files,
as are any tracebacks. Those records remain private. Shared-host contention and
sequential-seed temporal drift remain possible. Neither matching package hashes
nor matching versions establishes portable timing or equivalent host conditions.
