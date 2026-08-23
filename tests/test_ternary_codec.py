import unittest

from canaria.ternary_codec import (
    FIXED_V17_BYTES,
    decode_enumerative_v17,
    decode_fixed_v17,
    encode_enumerative_v17,
    encode_fixed_v17,
    pack_trits,
    unpack_trits,
)


class TernaryCodecTests(unittest.TestCase):
    def test_trit_roundtrip(self):
        values = [-1, 0, 1, 1, -1, 0, 0, 1, -1, 1, 0, -1]
        packed = pack_trits(values)
        self.assertEqual(unpack_trits(packed, len(values)), values)

    def test_fixed_v17_is_38_bytes_and_exact(self):
        support = [i % 4 for i in range(18)]
        trits = [(-1, 0, 1)[i % 3] for i in range(152)]
        scale = b"\x34\x12"
        stream = encode_fixed_v17(support, trits, scale)
        self.assertEqual(len(stream), FIXED_V17_BYTES)
        self.assertEqual(decode_fixed_v17(stream), (support, trits, scale))

    def test_enumerative_roundtrip_sparse(self):
        support = [(3 * i + 1) % 4 for i in range(18)]
        trits = [0] * 152
        for index in (0, 3, 9, 27, 51, 88, 120, 151):
            trits[index] = 1 if index % 2 else -1
        fixed = encode_fixed_v17(support, trits, b"\xaa\x55")
        compact = encode_enumerative_v17(fixed)
        self.assertLess(len(compact), len(fixed))
        self.assertEqual(decode_enumerative_v17(compact), fixed)

    def test_enumerative_roundtrip_dense(self):
        support = [0] * 18
        trits = [1 if i % 2 else -1 for i in range(152)]
        fixed = encode_fixed_v17(support, trits, b"\x00\x3c")
        compact = encode_enumerative_v17(fixed)
        self.assertEqual(decode_enumerative_v17(compact), fixed)


if __name__ == "__main__":
    unittest.main()
