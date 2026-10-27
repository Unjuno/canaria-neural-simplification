# Fixed scientific storage contract (source-only extract)

This is a portability description of the historical selected-cohort storage
algorithm, not a release token or extension of its completed execution authority.
Original full protocol and runtime identities are recorded in `SOURCE_MAP.json`.

## Frozen inputs and cohort

- Exactly 16 existing teacher IDs, 0–15; one teacher and previously selected C/J
  replacement state each: 48 source NPZ files. No selection/retry/codec search.
- All tensor values must be finite, little-endian float32. Every key and shape is
  checked. Source bytes must match SHA256; C/J logical hashes must also match.
- All 48 NPZ paths are distinct and relative to the explicitly provided input
  root. Inputs are read-only; no dataset or inference is needed.
- Selected replacement budgets come from the preexisting seven-budget grid:
  512, 1024, 1536, 2048, 3072, 4096, 6144. C width is budget/128; each of two J
  maps has width budget/256. This package does not choose any budget.

## Whole-model assembly

Teacher: stem Linear(64,64); four blocks each LayerNorm(64), Linear(64,128),
Linear(128,64); head Linear(64,10). There are 28 tensors and 71,626 parameters.

Whole C/J removes every `blocks.0.*` and `blocks.1.*` teacher tensor, retaining
stem, blocks 2/3 and head with original names and exact values. Replacement keys
are prefixed by `replacement.`. C has `w1.weight`, `w2.weight`; J has the same two
keys under each `0.` and `1.` prefix. No optimizer state, discarded front blocks,
or unused tensors are included. The unchanged portion has 38,218 parameters;
C/J total parameter count is 38,218 plus the selected replacement budget.
C has 18 tensors, J 20. All tensors in this particular schema are parameters.

## Serialization (unchanged scientific semantics)

For each model, sort tensor names and append `.npy`; write C-contiguous arrays
using NumPy NPY **v1.0**, `allow_pickle=False`. ZIP members use timestamp
1980-01-01 00:00:00, Unix `create_system=3`, `external_attr=0600<<16`.
No bundled metadata or member extras are added. Produce both ZIP_STORED and
ZIP_DEFLATED **level 6**, once each, with identical tensor payload/schema.
There are exactly 16 × 3 × 2 = **96** archives for a complete run.

Read every produced archive back with `allow_pickle=False`; verify keys, counts,
dtype, shape and exact raw array bytes. Archive byte counts are filesystem sizes,
not payload estimates. Separately record tensor payload bytes and parameter counts.
Source-to-destination mappings and array/archive SHA256 values are retained.

## Ratios, completeness and bounds

For parameters, stored archive bytes and deflated archive bytes, compute C/J,
C/teacher and J/teacher for every completed teacher. Aggregate with
`exp(sum(log(ratios))/n)` plus min, max, smaller/tie/larger. The denominator is the
whole compared model; replacement-only ratios must not be substituted.

A missing, corrupt or mismatched case remains a failed census entry; no cohort
replacement or retry is attempted. `cohort` remains 16 and `complete` explicitly
states the number of complete rows. Complete output requires 48 source records,
16 completed rows and 96 archives; a nonzero exit/incomplete state is not success.

The CLI preserves the one-process, 600-second alarm, 256 MiB address-space cap,
NumPy/stdlib-only scope and `OPENBLAS_NUM_THREADS=1`. Total source and produced
archive bytes must each remain below 128 MiB. Expanded ZIP input is additionally
bounded to 2 MiB per source, above the specified model schema but excluding
unbounded zip expansion. A hard interruption may leave incomplete output without
finished JSON; it is never implicitly resumed or reclassified as complete.
