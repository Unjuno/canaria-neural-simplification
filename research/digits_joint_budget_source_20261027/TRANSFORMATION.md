# Versioned source export and limits

## Preserved mathematics

Original grid runner SHA256:
`2c8e3a911b0eb25fd1b8434286dbc96f52e6d88829ee6e2de87e92c5da451bd2`.
Original PARENT.json SHA256:
`e4e25fd5be39987b259d3a641fdc47c4fb6a9cae152c9ca020a457e4bc4efc6c`.
Exact source/export digests and AST comparisons are in SOURCE_MAP.json.

`sha`, `load_npz`, `load_state`, `make_student`, `reused_cells` and `analyze`
retain their original ASTs. `run_seed` changes only source metadata and adds a
strict completed-parent scalar/seed check in place of the original two-field
status/ID check. Its teacher loading, data/split verification, targets, schedules,
fit order, optimizer calls, widths, final-only evaluation, 2048 scalar copying,
array outputs and fixed decision arithmetic are retained.

The mathematical helper is the exact pinned public matched package, not a copy
of another unexported helper. Its original source-to-public transformation is
already documented in that package. This grid export uses its public bounded
cohort/main lifecycle with grid namespace/token and explicit parent-input CLI
arguments. The original grid dynamically compiled private helper lifecycle
functions; the export includes the corresponding public lifecycle source
explicitly for inspectability. The original 19 tests are adapted/extended;
parent checks now live in a separate stdlib suite. Inclusion is not execution.

## Explicit nonhistorical custody change

The original finished attempt's private manifest, release, registry, ownership
messages and fixed infrastructure paths are not copied, reset or edited. Merely
stripping those binds would weaken the old verifier and would not close source
reproduction. Instead the export defines two new, incompatible interfaces:

- `canaria.grid-parent.v1`: a separately completed
  `canaria.matched-replication.v1` output and its entire private custody chain.
- `canaria.grid-replication.v1`: a separately authorized single grid attempt,
  exact source/dependency/parent descriptor hashes, finite deadline and one
  durable grid custody workspace, distinct from its parent's workspace.

This source package includes neither actual parent descriptor nor release. It
cannot attest global no-prior-attempt uniqueness across arbitrary workspace
copies; real custody/approval verification is the operator's responsibility.
Private records are checked at runtime but not distributed. Completed parent
approval may have expired: completion timestamps must fall within its original
run deadline; that does not renew the parent or authorize the grid.

Source package identifiers replace Git working-tree identities in generated
metadata. Original host/process metadata is not exported; future private
coordinator records still retain local process custody. Historical timing/output
bytes and source hashes are preserved as historical facts, not represented as
outputs of the portable launcher. No new test/data/model result is claimed.

## Scientific interpretation unchanged

This is still a sparse, fixed, adaptive, seen-data candidate grid. No monotonic
correction, continuous capacity claim, censored complete-case ratio, changed seed,
teacher replacement, retry, updated schedule, held-out test look or extra fit is
introduced by portability. The shared schedule and 600 updates are not equal
compute. Same-seed fresh replication does not provide new independent evidence.
