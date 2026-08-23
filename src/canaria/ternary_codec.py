"""Exact codecs extracted from the independently checked v17 core serialization.

These functions do not train a model.  They encode/decode the discrete state
of the v17 structured ternary core after quantization/support selection.

The fixed v17 format is:
    5 bytes  : eighteen 1-of-4 support indices (2 bits each)
    31 bytes : 152 ternary values, packed five trits per byte
    2 bytes  : opaque FP16 scale bytes supplied by the caller
    total    : 38 bytes

The enumerative format is lossless relative to that fixed format and exploits
zeros in the 152 ternary values.  It stores the nonzero count, combinadic rank
of nonzero positions, their sign bits, plus the same support and scale bytes.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

N_PATTERN_INDICES = 18
N_TRITS_V17 = 152
FIXED_V17_BYTES = 38


def _validate_trits(values: Sequence[int]) -> None:
    if any(v not in (-1, 0, 1) for v in values):
        raise ValueError("ternary values must be in {-1, 0, +1}")


def pack_trits(values: Iterable[int]) -> bytes:
    """Pack ternary values five per byte using digits 0,1,2 for -1,0,+1."""
    vals = [int(v) for v in values]
    _validate_trits(vals)
    out = bytearray()
    digits = [v + 1 for v in vals]
    for i in range(0, len(digits), 5):
        x = 0
        mul = 1
        for digit in digits[i : i + 5]:
            x += digit * mul
            mul *= 3
        out.append(x)
    return bytes(out)


def unpack_trits(buf: bytes, n_values: int) -> list[int]:
    """Inverse of :func:`pack_trits` when the number of values is known."""
    if n_values < 0:
        raise ValueError("n_values must be non-negative")
    values: list[int] = []
    for byte in buf:
        x = int(byte)
        for _ in range(5):
            values.append((x % 3) - 1)
            x //= 3
    if len(values) < n_values:
        raise ValueError("buffer is too short for requested ternary count")
    return values[:n_values]


def pack_pattern_1of4(indices: Iterable[int]) -> bytes:
    """Pack the v17 eighteen 1-of-4 support indices into five bytes."""
    idx = [int(v) for v in indices]
    if len(idx) != N_PATTERN_INDICES:
        raise ValueError(f"expected {N_PATTERN_INDICES} support indices")
    if any(v < 0 or v > 3 for v in idx):
        raise ValueError("each support index must be in [0, 3]")
    x = 0
    for i, value in enumerate(idx):
        x |= (value & 0x3) << (2 * i)
    return x.to_bytes(5, "little")


def unpack_pattern_1of4(buf: bytes) -> list[int]:
    """Inverse of :func:`pack_pattern_1of4`."""
    if len(buf) != 5:
        raise ValueError("v17 pattern buffer must be exactly five bytes")
    x = int.from_bytes(buf, "little")
    return [int((x >> (2 * i)) & 0x3) for i in range(N_PATTERN_INDICES)]


def encode_fixed_v17(
    support_indices: Sequence[int], trits: Sequence[int], scale_f16_bytes: bytes
) -> bytes:
    """Encode the exact 38-byte fixed v17 core representation."""
    if len(trits) != N_TRITS_V17:
        raise ValueError(f"expected {N_TRITS_V17} ternary values")
    _validate_trits(trits)
    if len(scale_f16_bytes) != 2:
        raise ValueError("scale_f16_bytes must contain exactly two bytes")
    out = (
        pack_pattern_1of4(support_indices)
        + pack_trits(trits)
        + bytes(scale_f16_bytes)
    )
    if len(out) != FIXED_V17_BYTES:
        raise AssertionError(f"fixed v17 codec produced {len(out)} bytes")
    return out


def decode_fixed_v17(buf: bytes) -> tuple[list[int], list[int], bytes]:
    """Decode the exact fixed v17 representation."""
    if len(buf) != FIXED_V17_BYTES:
        raise ValueError(f"fixed v17 stream must be {FIXED_V17_BYTES} bytes")
    support = unpack_pattern_1of4(buf[:5])
    trits = unpack_trits(buf[5:36], N_TRITS_V17)
    scale = bytes(buf[36:38])
    return support, trits, scale


def _rank_combination(positions: Sequence[int]) -> int:
    """Combinadic rank used by the historical v17 enumerative codec."""
    return sum(math.comb(int(pos), i + 1) for i, pos in enumerate(positions))


def _unrank_combination(rank: int, k: int, n: int) -> list[int]:
    if k < 0 or k > n:
        raise ValueError("invalid combination size")
    if rank < 0 or rank >= math.comb(n, k):
        raise ValueError("combination rank out of range")
    positions = [0] * k
    remainder = rank
    x = n - 1
    for i in range(k, 0, -1):
        while math.comb(x, i) > remainder:
            x -= 1
        positions[i - 1] = x
        remainder -= math.comb(x, i)
        x -= 1
    return positions


def _pack_bits(values: Sequence[int]) -> bytes:
    if any(v not in (0, 1) for v in values):
        raise ValueError("bit values must be 0 or 1")
    x = 0
    for i, value in enumerate(values):
        x |= int(value) << i
    return x.to_bytes((len(values) + 7) // 8, "little")


def _unpack_bits(buf: bytes, n_values: int) -> list[int]:
    x = int.from_bytes(buf, "little")
    return [int((x >> i) & 1) for i in range(n_values)]


def encode_enumerative_v17(fixed_stream: bytes) -> bytes:
    """Losslessly compress a 38-byte fixed v17 stream using zero positions.

    This reproduces the historical Phase-W format.  Its size is data dependent.
    """
    support, trits, scale = decode_fixed_v17(fixed_stream)
    nonzero_positions = [i for i, value in enumerate(trits) if value != 0]
    k = len(nonzero_positions)
    if k > 255:
        raise ValueError("v17 format stores k in one byte")
    rank = _rank_combination(nonzero_positions)
    combinations = math.comb(N_TRITS_V17, k)
    rank_bits = 0 if combinations <= 1 else math.ceil(math.log2(combinations))
    rank_bytes = (rank_bits + 7) // 8
    signs = [1 if trits[i] > 0 else 0 for i in nonzero_positions]

    out = bytearray(pack_pattern_1of4(support))
    out.append(k)
    out.extend(rank.to_bytes(rank_bytes, "little"))
    out.extend(_pack_bits(signs))
    out.extend(scale)
    return bytes(out)


def decode_enumerative_v17(buf: bytes) -> bytes:
    """Decode a Phase-W enumerative stream back to the 38-byte fixed stream."""
    if len(buf) < 8:  # pattern + k + scale is the absolute minimum
        raise ValueError("enumerative stream is too short")
    support_bytes = bytes(buf[:5])
    k = int(buf[5])
    if k > N_TRITS_V17:
        raise ValueError("invalid nonzero count")
    combinations = math.comb(N_TRITS_V17, k)
    rank_bits = 0 if combinations <= 1 else math.ceil(math.log2(combinations))
    rank_bytes = (rank_bits + 7) // 8
    sign_bytes = (k + 7) // 8
    expected = 6 + rank_bytes + sign_bytes + 2
    if len(buf) != expected:
        raise ValueError(f"expected {expected} bytes for k={k}, got {len(buf)}")

    pos = 6
    rank = int.from_bytes(buf[pos : pos + rank_bytes], "little")
    pos += rank_bytes
    signs = _unpack_bits(bytes(buf[pos : pos + sign_bytes]), k)
    pos += sign_bytes
    scale = bytes(buf[pos : pos + 2])

    nonzero_positions = _unrank_combination(rank, k, N_TRITS_V17)
    trits = [0] * N_TRITS_V17
    for sign, index in zip(signs, nonzero_positions):
        trits[index] = 1 if sign else -1

    support = unpack_pattern_1of4(support_bytes)
    return encode_fixed_v17(support, trits, scale)
