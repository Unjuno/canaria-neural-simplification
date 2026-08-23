"""Small reusable components extracted from the Canaria research codebase.

Historical experiment scripts remain under ``scripts/``.  Modules under
``src/canaria`` are intentionally dependency-light, cleaned interfaces for
reusing methods without depending on the original experiment filesystem.
"""

from .ternary_codec import (
    FIXED_V17_BYTES,
    N_PATTERN_INDICES,
    N_TRITS_V17,
    decode_enumerative_v17,
    decode_fixed_v17,
    encode_enumerative_v17,
    encode_fixed_v17,
    pack_pattern_1of4,
    pack_trits,
    unpack_pattern_1of4,
    unpack_trits,
)

__all__ = [
    "FIXED_V17_BYTES",
    "N_PATTERN_INDICES",
    "N_TRITS_V17",
    "decode_enumerative_v17",
    "decode_fixed_v17",
    "encode_enumerative_v17",
    "encode_fixed_v17",
    "pack_pattern_1of4",
    "pack_trits",
    "unpack_pattern_1of4",
    "unpack_trits",
]
