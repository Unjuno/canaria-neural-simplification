from __future__ import annotations

import unittest

from scripts.gaussian_shift_interface.evaluate_c65r import evaluate


class C65REvaluatorTests(unittest.TestCase):
    def row(self, seed, p0_acc=.950, p2_acc=.952, p0_nmse=.100, p2_nmse=.098):
        return {
            "seed": seed,
            "eligible": True,
            "p0_validation_accuracy": p0_acc,
            "p2_validation_accuracy": p2_acc,
            "p0_nmse": p0_nmse,
            "p2_nmse": p2_nmse,
        }

    def test_confirmatory_pass(self):
        rows = [self.row(s) for s in range(63400, 63416)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "C65R_CONFIRMATORY_PASS")
        self.assertTrue(out["validation_noninferiority"]["pass"])
        self.assertTrue(out["nmse_ratio"]["pass"])

    def test_validation_fail(self):
        rows = [self.row(s, p0_acc=.90, p2_acc=.95) for s in range(63400, 63416)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "C65R_CONFIRMATORY_FAIL")
        self.assertFalse(out["validation_noninferiority"]["pass"])

    def test_nmse_fail(self):
        rows = [self.row(s, p0_nmse=.14, p2_nmse=.10) for s in range(63400, 63416)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "C65R_CONFIRMATORY_FAIL")
        self.assertFalse(out["nmse_ratio"]["pass"])

    def test_insufficient_eligible(self):
        rows = [self.row(s) for s in range(63400, 63407)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "STOP_INSUFFICIENT_ELIGIBLE")

    def test_unexpected_seed_rejected(self):
        rows = [self.row(s) for s in range(63400, 63416)]
        rows.append(self.row(99999))
        with self.assertRaises(ValueError):
            evaluate({"rows": rows})


if __name__ == "__main__":
    unittest.main()
