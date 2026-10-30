# Analysis-only transformation

The original source files combine scientific calculation with training,
serialization, timing, execution lifecycle and environment-specific coordination.
This new module transcribes only saved-scalar arithmetic into standalone functions.
It neither imports nor executes those original modules. All execution/environment
configuration, dataset acquisition, training, model reconstruction, tensor
serialization, live measurements and lifecycle coordination are omitted.

Original commit/hash references are in SOURCE_PROVENANCE.json. Matched/grid
arithmetic follows their analyze functions; storage follows ratio aggregation;
latency follows summarize. Constants and bootstrap RNG/draw order are preserved.
Storage intentionally uses sum(log)/n as the original; latency uses statistics.mean.
Matched and grid rows are sorted by seed ID before bootstrap.

Additional defensive validation rejects malformed records and duplicate identities
instead of allowing accidental complete-case analysis. The new public API raises
InvalidInput (or missing-key errors), rather than mirroring execution supervisor
failure records. Optional matched accuracy values are consistency-checked. No
source execution is implied by recalculating scalar summaries.

The comparison command checks existing published-candidate scalar tables using exact
Python equality against summary structures; synthetic tests exercise failures.
These are implementation tests, not new scientific trials or experimental retries.
