# Portability transformation and evidence boundary

Origin: the historical `research/selected_whole_storage/measure.py` identified
by SHA256 in `SOURCE_MAP.json`; the original was read only and remains untouched.
This is a versioned derivative, not a byte-identical launcher or a newly completed
historical experiment. Modified files carry that notice in their module headers.

## Preserved scientific operations

- Exact ASTs of `shapes_teacher` and `state_hash` are extracted into
  `storage_codec.py` with no semantic change.
- Serializer writer body retains sorted tensor names, C-contiguous arrays,
  NPY v1.0, `allow_pickle=False`, fixed ZIP timestamp/Unix metadata, stored and
  DEFLATE level 6. Only output open mode changes from overwrite `w` to exclusive
  `x`; explicit readback exceptions replace optimization-sensitive assertions.
- Whole-model assembly preserves teacher stem/blocks2–3/head, removes blocks0–1,
  prefixes selected C/J keys and retains the entire model denominator.
- 16 teachers, 48 source/model states, two encodings and 96 output archives;
  actual file sizes, tensor payload bytes/counts, raw tensor mapping hashes,
  C/J, C/teacher, J/teacher ratios, geometric summaries and failure census.

## Deliberate non-scientific changes

- Remove import-time `results.mkdir(exist_ok=False)`, alarm, resource limit,
  numerical environment mutation and cohort execution. `main()` guards execution;
  required CLI paths replace hardcoded/local source and result paths.
- New portable v1 selection contract rather than copying the private historical
  selection table. Preserve choices through a reviewed custodian handoff, not by
  pretending a rewritten manifest equals the original. Pin relocated manifest
  bytes explicitly. Source identities/budgets remain required inputs.
- Fail before manifest/numerical loading on output collisions. Atomic absent-only
  mkdir and exclusive files prevent reuse/overwrite; no directory is removed.
- Strict JSON and path validation; input symlink rejection; source bytes are
  hashed and decoded from the same in-memory byte string. Bound ZIP expansion and
  input/artifact totals. This tightens invalid-input behavior without changing
  the permitted float32 model schema or archive payloads.
- Replace host/Git/clock/process metadata with portable version and source hashes.
  Sanitized failure records carry exception types, not arbitrary path-bearing text.
- Add exact historical-version gate and compile/runtime zlib identity check.
  Preserve the 600-second/256-MiB/one-process scope in the CLI, not on import.
- Add explicit COMPLETE/INCOMPLETE and expected/produced archive accounting.
  No partial teacher enters completed-row ratio aggregation.
- Supply independently written artificial-only tests. They do not open original
  weights, validation data or historical archives; package creation did not run
  those tests or import the candidate.

## Evidence levels

AST parsing, hashes, import inventories, literal/schema/path checks and source
comparison establish only static inspectability. They do not establish actual
serialization determinism, historical result equality, full upstream selection
reconstruction, runtime compatibility, execution permission or publication of
payloads. Separate reviewer artificial tests can establish a limited algorithm
check, not an independent scientific replication. No latency extension, new
framework, changed selection rule or compression search is included.
