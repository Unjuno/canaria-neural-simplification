# Phase W — Exact enumerative codec for zero-heavy ternary coefficients

No model values are changed. Starting from the exact 38-byte Phase V codec for the independently confirmed 44.5-byte nominal representation:
- keep the 5-byte support-pattern field and 2-byte FP16 scale;
- among N=152 ternary coefficient symbols, store k = number of nonzeros in one byte;
- store the set of k nonzero positions by its exact combinatorial (colex) rank among C(152,k) subsets;
- store one sign bit for each nonzero coefficient.

Decoder must exactly reconstruct all 152 trits, pattern bytes and scale bytes. Report actual byte length per seed. No performance inference is needed because decoded model is bit-exact to Phase V.
