# Phase V — Exact ternary serialization of the 44.5 B nominal model

The signed 2-bit quantizer used in Phases N–U has qmax=1 and therefore emits only three levels {-scale, 0, +scale}. This experiment tests an exact fixed codec rather than changing the model.

Codec:
- 152 ternary stored values encoded 5 trits per byte: ceil(152/5)=31 bytes.
- 18 shared 1-of-4 support indices encoded at 2 bits each: 36 bits -> 5 bytes.
- one FP16 scale: 2 bytes.
- total fixed byte stream: 38 bytes.

Reconstruct and require bit-exact support/trits, max FP32 reconstructed-weight difference = 0, and identical model predictions/accuracy before versus after serialization. Evaluated on the 8 independent Phase U seeds 2700–2707.