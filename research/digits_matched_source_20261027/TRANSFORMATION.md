# Explicit source transformation, version 1

This is a coherent source-only port for a **new, independently authorized matched
replication**, not a byte-identical export of the historical operational package.
`SOURCE_MAP.json` records original and export file/definition SHA256 values.

| Component | Transformation / preservation |
|---|---|
| Historical helper | Extract only the exact source text of `ResBlock`, `Net`, `TinyRes`, `train_model`; add minimal imports/docstring. The original helper's full file SHA and public Git blob remain provenance. Runtime verifies the new extracted-file SHA, then preserves the four-definition AST allowlist. |
| Numerical science | Preserve clone/isolation, initialization, schedule, local/joint/compositional losses, optimizer settings, scoped sqrt intervention, teacher recipe, splitting, data scaling, endpoint propagation, final checkpoints, metrics and bootstrap arithmetic. No threshold, seed, budget or failed-cell changes. |
| Metadata-only scientific changes | `run_seed` startup metadata records package identity instead of repository HEAD/PID. Environment records hash raw torch build text instead of exposing compiler paths. Analyzer scope text says fixed-seed regimen, not new randomness for a repeat of already-used seeds. |
| Package identity | Replace repository HEAD/clean-worktree checks with an exact source/test/schema/environment file manifest and canonical manifest digest. This works without a Git checkout and binds the complete supplied runtime dependency closure. It is not the original Git commit identity. |
| Private registry | Remove the author's absolute historical registry path. Introduce `canaria.matched-replication.v1`: one declared new identity in one durable operator-owned workspace, fixed relative custody/output layout, exclusive permanent mkdir, no automatic retry, and explicit historical-closed attestation. No historical registry/token/release bytes are copied, read at runtime, reset or bypassed. |
| Approval/deadline | Keep actual approval and independent GO requirements, now bound to exported package hashes and a new replication's finite deadline. The historical 2026-10-29 06:50 UTC deadline is retained as historical protocol text, **not extended**. A separately authorized new replication has its own deadline, capped to 3 hours elapsed and 600 seconds per seed. No valid release is supplied. |
| Private versus scientific output | Full approvals/tokens/process coordination stay under private custody; new scientific output carries only the reduced `run_contract.json`. The private coordinator still binds token, output location and end time. Runtime logs can contain local diagnostics and are not automatically publishable. |
| Lifecycle safeguards | Keep two-worker limit, RSS/evidence/time caps, bounded TERM/KILL cleanup, complete nonzero/spawn/failure/not-started census, exclusive seed directories and all-16 completeness gates. Adapt fake-process tests to the new context, without weakening the assertions. |
| Tests | Preserve original synthetic numerical checks; adapt hash/release/lifecycle fixtures to the explicit contract. Add 12 stdlib-only contract checks. Fixtures have expired deadlines and are not usable authorization. No supplied test was run during packaging. |
| Protocol/seeds | Seeds are byte-identical. Use the already-public scientific protocol extract, changing only its container-specific provenance pointer to this package's SOURCE_MAP/TRANSFORMATION. Original private preparation/coordination records are not included. |
| Environment | Add all observed runtime dependency versions including two transitive pins absent from the old hand-written requirements. Remove operational host/compiler paths from documentation. No wheel/interpreter/OS archive is included and no installation was performed. |

## Custody trust boundary

A filesystem marker can coordinate launchers sharing the same durable root; it
cannot prevent a human from cloning software or inventing another root/identity.
The contract therefore explicitly requires an accountable custodian to attest
that the declared replication has no prior attempt and all launchers use one
root. A failed attempt cannot be relabeled as another replication to obtain a
success. Any genuinely different experiment requires a separate protocol and
authority; this source package does not decide or grant it. The original local
consumed attempt is unaffected and is never a runnable release for this port.

This transformation does not make the missing original raw artifacts public,
validate original tensors, establish new data independence, reproduce the grid's
private parent-manifest verification, or turn scientific generation into a
saved-scalar-only operation. All numerical/library/runtime tests remain for a
separately authorized reviewer.
