# Seven-budget J/C grid: portable source-only candidate

**Static inspection only; execution validation is pending.** This package supplies
source, strict parent/release contracts and synthetic tests. It contains no data,
models, arrays, historical custody records, actual parent descriptor or execution
release. Packaging did not import project code or run tests, fits, inference,
installs, downloads, CI, measurements or scientific attempts.

## What is closed, and what is not

The original accepted grid is complete at
`dbbce006c07e703376dfe7cd0832e5bb3bd2c584` (runtime
`4c9e277d5f1ee913a8a75f1eadeb5c6f0ab9e96a`). Its recorded result was
`CONDITIONAL_GRID_ADVANTAGE`: geometric first-observed-pass C/J budget ratio
0.788324, paired-bootstrap 95% interval [0.694450, 0.904368]. That historical result
is not a result of this source package. See the
[published historical grid report](https://github.com/Unjuno/canaria-neural-simplification/blob/1de2b5f046ef2219270af2e029c5479a3eeee10a/research/public_checkpoint_20261027/research/digits_joint_budget_grid/results/REPORT.md).

The original private parent chain is **not** exported or validated here. A new,
separately authorized replication can instead consume a complete fresh output of
the pinned [matched source package](../digits_matched_source_20261027/README.md).
That is a new execution on the same fixed seeds and already-seen Digits, **not**
new independent seeds/data, historical archive recovery or a reopened attempt.
No output is guaranteed numerically or byte-identical across hosts.

## Exact dependency closure

Keep these two directories as siblings under `research/`:

```text
digits_matched_source_20261027/       # all 18 pinned public source/doc/test files
digits_joint_budget_source_20261027/ # this package
```

`DEPENDENCY.json` pins every file in the co-published matched sibling by SHA256,
and separately records the matched release-required file map/package identity.
Its `commit` is the original publication provenance, not the amended bytes;
`amendment_base_commit` identifies the reviewed base for the finalization-status
repair and added fake-clock regression. Keep the amended siblings together.
`parent_contract.dependency()` verifies
all pinned bytes before numerical import. The grid imports the matched restricted
runner, which compiles only the four allowlisted scientific definitions. There
is no dependency on the absent original matched/grid helpers, repository Git
state, another local checkout, private paths or historical release files.

Runtime dependencies and exact provisioning pins are inherited without changes;
see `ENVIRONMENT.md`. This is source closure plus a version recipe, **not** a
hermetic binary/OS lock or proof that the recorded wheels remain obtainable.

## Fixed science retained

- Seven budgets: **512, 1024, 1536, 2048, 3072, 4096, 6144** weights.
- J: two maps of width h; C: one map of width 2h. Original fixed grid seeds copied
  byte-identically. Original fit/validation split and teacher seeds come from the
  pinned matched parent; fit 1,077 / validation 270, stratification RNGs 1234/5678.
- No teacher fit and no 2048 refit in the grid. Reuse 32 J/C parent cells verbatim;
  fit 16 teachers × 6 missing budgets × 2 arms = **192** replacements.
- All **224** cells required. Shared parent 600×128 schedule, exactly 600 updates,
  explicit AdamW settings, scoped float64-square-root profile, final weights only.
  Every teacher's twelve fits finish before new validation inference.
- Pass iff NMSE ≤ .08 and accuracy ≥ same-teacher accuracy − .02. Retain failed
  cells and nonmonotonic patterns; the first passing observed candidate is **not a
  theoretical/continuous minimum**. No monotonic interpolation or adaptive width.
- Ratio/100,000-draw paired bootstrap (seed 2710276144) only if both arms pass
  somewhere for **all 16** teachers. Otherwise `CENSORED_NO_COHORT_RATIO`, never a
  complete-case ratio or invented infinity. Missing/failed cohort is `INCOMPLETE`.

Equal weights and updates do not imply equal compute. This adaptive follow-on on
seen teachers/validation data is not fresh-data confirmation, objective-only
causality, a universal architecture result, compression or deployment speed.

## Reader checks, only after separate execution review

From this directory in the two-sibling layout:

```sh
python test_parent_contract.py
```

The standard-library tests manufacture opaque bytes and JSON, not usable model
or dataset payloads. They exercise the complete 339+19-file contract, nested
manifests, custody completion, source/release binding and refusal cases.

In the recorded numerical environment, a separate reviewer may approve:

```sh
python test_runner.py
```

These tests use generated tensors, small synthetic forwards/backwards/optimizer
steps, scalar grids and fake subprocesses. They must not call a dataset loader,
load real models, spawn a real scientific process or run a cohort. Tests are
**supplied, not newly passed**. `STATIC_VALIDATION.json` is not a runtime test log.

## Future replication workflow — requires separate actual authorization

1. Obtain independent review and authorization for a new matched replication;
   complete it using the pinned matched package in its one durable custody root.
   No matched execution is authorized by this document.
2. After completion, build a **private** descriptor without decoding any arrays:

   ```sh
   python parent_contract.py --workspace MATCHED_CUSTODY_ROOT --replication-id MATCHED_REPLICATION_ID --out PRIVATE_PARENT.json
   ```

   This hashes all 339 outputs plus 19 private custody records and verifies full
   manifests, all 16 successful exits, source/seed identities, scalar 2048 reuse,
   permanent claim, private approval and completion/deadline cross-bindings. It
   does not import ML packages, evaluate models, issue approval or repair a parent.
3. Obtain separate grid review and actual authorization. Bind the descriptor SHA,
   parent replication identity, matched package identity, all `REQUIRED_FILES`
   hashes, a new grid identity and finite UTC deadline in a private release
   conforming to `RELEASE_SCHEMA.json`. String assertions are not signatures;
   the custodian must verify genuine approval and absence of a prior attempt.
4. Only under that authorization, choose a separate durable grid workspace and:

   ```sh
   python runner.py cohort --workspace GRID_CUSTODY_ROOT --parent-workspace MATCHED_CUSTODY_ROOT --parent-manifest PRIVATE_PARENT.json --release PRIVATE_GRID_RELEASE.json
   ```

The grid verifies the complete parent before consuming its permanent attempt and
again before/after each seed. It atomically claims
`custody/grid/<replication_id>/` before output creation; no reset/resume/retry flag.
Never delete/rename the registry or choose another identity/root to retry the same
approved replication. Two workers maximum; 600 seconds/seed; 3 hours/cohort;
4 GiB RSS/worker; 1 GiB output evidence; bounded TERM/grace/KILL cleanup. All
failures/not-started IDs remain in the census. Outputs go to
`artifacts/<replication_id>/` and require privacy review before any distribution.

Historical deadlines and registries remain unchanged, closed and unconsulted.
This interface does not change any external-data acquisition or held-out-test
hold. Original raw archive distribution, external replication, storage/latency
source closure and scientific generalization remain separate open matters.
