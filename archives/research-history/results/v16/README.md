# v16 results index

Phase L/M replaced unstructured sparse support with structured support and then re-tested selected conditions on an independent cohort.

Protocols/results:
- [`../../docs/phases/v16/37_PHASEL_STRUCTURED_SPARSITY_PROTOCOL_V16.md`](../../docs/phases/v16/37_PHASEL_STRUCTURED_SPARSITY_PROTOCOL_V16.md)
- [`../../docs/phases/v16/38_PHASEM_STRUCTURED_HOLDOUT_PROTOCOL_V16.md`](../../docs/phases/v16/38_PHASEM_STRUCTURED_HOLDOUT_PROTOCOL_V16.md)
- [`../../docs/phases/v16/39_PHASEL_M_STRUCTURED_RESULTS_V16.md`](../../docs/phases/v16/39_PHASEL_M_STRUCTURED_RESULTS_V16.md)

Key conclusions:
- structured support reduced index overhead enough to beat the dense 3-bit storage frontier at useful fidelity;
- 2:4 × 3-bit was the most reproducible no-repair condition across the exploration and independent holdout cohorts;
- more aggressive ~108 B kernel-block representations were less stable without repair but recovered to roughly 0.97 matched-control utility after short repair in two independent cohorts;
- these findings motivated the explicit sub-100-byte sequence in v17.
