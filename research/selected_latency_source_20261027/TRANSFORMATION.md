# Source transformation and limits

Original accepted latency source commit:
`09efa9d7323c696cc0a5a73d5361d053d34a0319`.
Public source-only work is based on research checkpoint
`6b27220ada063ae1362e8efd412ecfeae238fc42`.
Source identities/export digests are in SOURCE_MAP.json. Original manifests and
private operational paths are not exported; their source-byte hashes identify
provenance without releasing contents.

## Exact scientific definitions

AST-identical extractions: Deploy, ResBlock, TinyRes, ordered, timed_block,
summarize, EvidenceWriter and failure_census. Only the needed two base classes
are vendored; no Net/training/full-dataset runner is imported or executed.
The original literal permutation schedule is preserved, not alphabetically sorted.
The scientific timing loop, synthetic input formula, warmups, iterations, cohort,
no_grad, all-model-before-timing correctness gate and median/geometric aggregation
retain original behavior. All outliers remain; no scientific result is altered.

## Explicit operational differences

- Remove numerical imports/runtime configuration and dynamic AST execution from
  control-module import. The separately launched worker imports a static graph file.
- Replace embedded private paths and original registry/release with a versioned,
  explicit contained-input contract and separately approved fresh private attempt.
  Reject retries/overwrites and bind the new control-root identity in the release.
- Retain exact original choice/reference identities through hash-bound private
  provenance files, without opening their embedded old paths. Vendored storage
  selection validator is byte-identical to the companion storage source package.
- Preserve the loader graph/tensor checks and 256+14 validation numerics; add integer
  split shape/range checks, strict raw-row schedule/census checks and reject `-O`.
- Bind complete source contents and clean reviewed commit, not the old consumed
  operational manifest. A new schema is not historical release verification.
- Supervisor claims the permanent attempt before worker imports and owns both
  leases. It also records launch/native failures, malformed metadata and explicit
  non-completion, and handles exceptional postlaunch termination/reaping.
- Keep 590+5+5s limits, 4GiB worker address limit and 128MiB evidence budget/reserves;
  numerical imports now occur inside that memory limit, a stricter setup gate.
- Keep raw environment/process captures and failure tracebacks in conspicuously
  PRIVATE generated files. No such records or payload files are supplied publicly.
- Remove preparation inference entry point. Tests use only fake callables/clocks
  and artificial metadata; importing the tests never imports the deployment module.

## What static review does not establish

No code was imported/executed; no test passed claim, numerical result, timing,
scientific replay or environment setup is asserted. AST equality proves selected
source definitions match; it does not prove runtime dependency behavior, correct
payload custody, numerical equivalence on another machine, safety of arbitrary
external files or portable historical latency. A fresh independent static review
and separately authorized synthetic controls precede any consideration of execution.
No CI, acquisition, paid service, remote write or upstream research change occurred.
