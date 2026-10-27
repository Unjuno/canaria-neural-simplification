# Matched L/J/C: portable source-only candidate

**Static checks passed; execution validation is pending.** This package
closes the restricted matched runner's source dependency gap. It does **not**
supply the original models, arrays, private execution records or an approval to
run. No package code was imported, no tests were run and no training/inference,
data download, installation, CI or publication occurred during packaging.

The historical matched study is finished. A separately authorized outside
replication is possible using the explicit portable contract below; it is a new
execution of the **same fixed seeds on already-seen Digits**, not new independent
seeds, new data or a reopening of the original consumed attempt.

## Contents

- `runner.py`: restricted teacher + matched L/J/C implementation and bounded
  cohort coordinator, transformed from original runner SHA256
  `f9faf359eec05c05ed1e23464160514c409c7eecd578d5e87ffa3fc819ad1b7e`.
- `scientific_definitions.py`: exact `ResBlock`, `Net`, `TinyRes`, `train_model`
  definitions from the pinned historical source. No historical dataset/full-grid
  runner or test evaluation is included or compiled.
- `SEEDS.json`, `PROTOCOL.md`: fixed 16 seed records and published historical
  scientific protocol extract. Historical preparation tenses are not current
  authorization; see `TRANSFORMATION.md` for the explicit portability changes.
- `test_runner.py`: 15 adapted historical synthetic checks, including fake-process
  lifecycle tests. `test_portable_contract.py`: 12 new stdlib contract checks.
  **27 test methods supplied, none executed for this candidate.**
- `portable_contract.py`, `RELEASE_SCHEMA.json`: matched-specific fresh-replication
  authorization/custody interface. **No release is included.**
- `ENVIRONMENT.md`, `requirements.lock.txt`: exact recorded version/numerical
  contract and future provisioning recipe, not a hermetic binary/OS lock.
- `ARTIFACT_SCHEMA.json`, `ARTIFACTS.md`: relative paths, arrays/state inventory,
  hash semantics and private/public custody boundary, with no payload bytes.
- `SOURCE_MAP.json`, `STATIC_VALIDATION.json`: source/export digests and static
  AST/dependency checks, not runtime validation.

## Scientific contract retained

Four-block 64-dimensional teacher, original 60-epoch recipe; replace its first
two blocks. Fit 1,077 / validation 270, original stratified RNGs 1234 and 5678,
features scaled by 16. Full bundled data/labels are read for stratification, but
test feature/label subsets are never materialized or evaluated.

Each arm has 2,048 learned weights and consumes the same 600×128 with-replacement
schedule. L/J share exact deep-cloned initial two-map tensors (width 8 each).
L sums the two teacher-forced local MSEs; J jointly minimizes endpoint MSE.
C is one width-16 map, separately initialized, and only an architecture comparator.
All replacements use the explicit AdamW flags and scoped float64-square-root
intervention. Validation happens after every arm has finished, at final step 600;
there is no best-checkpoint or budget selection.

All 16 IDs are required. The original bootstrap, seed 271027042, 100,000 draws,
feature gate, accuracy noninferiority gate and quality thresholds are retained.
Same updates/example exposures do not establish equal FLOPs or wall time. The
training-regime effect does not isolate objective, teacher forcing and gradient
path mechanisms. No C comparison rescues a failed J/L primary.

## Reader validation, only after separate execution review

Source inspection needs only a text reader. From this package directory, a
reviewer may separately approve the standard-library synthetic checks:

```sh
python test_portable_contract.py
```

In the recorded numerical environment, the separate tensor checks are:

```sh
python test_runner.py
```

These tests use generated tensors and fake subprocesses, not real Digits or
teacher training. Tensor tests do perform small synthetic forwards/backwards and
optimizer steps. Do not interpret their mere inclusion as a passing test result.
Importing `runner.py` configures numerical libraries and compiles allowlisted
definitions; it does not itself load data or start a cohort.

## New-replication contract, not historical-run authority

A researcher must obtain independent review and actual authorization, predeclare
one new `replication-...` identity, bind the exact `REQUIRED_FILES` hashes and
package identity, choose a finite UTC deadline, and establish **one durable custody
workspace shared by all launchers for that approved replication**. The release
is private and must conform to `RELEASE_SCHEMA.json` plus runtime validation.
String fields are provenance assertions, not cryptographic signatures: the
custodian remains responsible for verifying the real approvals and no prior
attempt. This source package cannot issue that authority.

For such an independently approved replication only, the entry point is:

```sh
python runner.py cohort --workspace DURABLE_CUSTODY_WORKSPACE --release APPROVED_RELEASE.json
```

The workspace must already exist; source and result paths are kept separate.
There is no dataset-download path. The coordinator atomically creates
`custody/matched/<replication_id>/` **before** creating output, permanently
consuming the attempt even if later setup or spawning fails. Outputs go to
`artifacts/<replication_id>/`. Never delete/rename the registry, choose another
workspace/identity, or change a deadline to retry this replication. There is no
resume/reset/retry flag. Child invocation is token-bound; seed output creation is
exclusive. Maximum two isolated single-thread seed processes, 600 seconds/seed,
3 hours/cohort, 4 GiB RSS/worker, 1 GiB persisted evidence and the earlier approved
deadline are retained, with bounded TERM/KILL cleanup and complete failure census.

The original deadline and consumed registry are **not changed or consulted** by
this separate replication interface. That separation is an explicit versioned
contract transformation, not a claim that the original registry was exported or
that a cloned directory can establish globally unique custody. No local current
or historical scientific attempt was created by packaging.

## Limits and existing public evidence

Fresh outputs would be new artifacts, not substitutes for the original missing
archive. Equal pins do not guarantee cross-host byte identity. Raw artifact
licensing/custody/distribution, runtime portability, original grid parent chains,
storage and latency remain separate gaps. This package is matched-only and does
not extend to those runners. It does not access external-writer data or change
any existing acquisition or holdout gate. Generated error logs and private
approval records require separate publication review.

Pinned public baseline:
[`d42dc542a433d3a3554a0f63abe0794a033890c5`](https://github.com/Unjuno/canaria-neural-simplification/tree/d42dc542a433d3a3554a0f63abe0794a033890c5).
The [historical helper](https://github.com/Unjuno/canaria-neural-simplification/blob/d42dc542a433d3a3554a0f63abe0794a033890c5/scripts/reproduce/core_discovery_digits/run_confirmatory.py)
is not a substitute for the restricted runner. The
[published matched results](https://github.com/Unjuno/canaria-neural-simplification/blob/d42dc542a433d3a3554a0f63abe0794a033890c5/research/public_checkpoint_20261027/research/digits_matched_regime/results/REPORT.md)
and [saved-scalar analysis](https://github.com/Unjuno/canaria-neural-simplification/blob/d42dc542a433d3a3554a0f63abe0794a033890c5/research/public_saved_analysis_20261027/README.md)
remain the historical evidence; this source candidate supplies no new result.
