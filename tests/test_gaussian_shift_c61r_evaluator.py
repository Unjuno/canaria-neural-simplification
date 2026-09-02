import unittest

from scripts.gaussian_shift_interface.evaluate_c61r import evaluate


class C61REvaluatorTests(unittest.TestCase):
    @staticmethod
    def rows(n=16, val_diff=-0.005, ratio=1.05):
        out = []
        for i in range(n):
            p8_acc = 0.95
            p8_nmse = 0.02
            out.append({
                "seed": 59400 + i,
                "eligible": True,
                "p4_validation_accuracy": p8_acc + val_diff,
                "p8_validation_accuracy": p8_acc,
                "p4_nmse": p8_nmse * ratio,
                "p8_nmse": p8_nmse,
                "frozen_nmse": 0.03,
            })
        return out

    def test_pass(self):
        r = evaluate(self.rows())
        self.assertEqual(r["decision"], "C61R_CONFIRMATORY_PASS")
        self.assertTrue(r["validation_noninferiority"]["pass"])
        self.assertTrue(r["nmse_ratio"]["pass"])

    def test_validation_fail(self):
        r = evaluate(self.rows(val_diff=-0.03))
        self.assertEqual(r["decision"], "C61R_CONFIRMATORY_FAIL")
        self.assertFalse(r["validation_noninferiority"]["pass"])

    def test_nmse_fail(self):
        r = evaluate(self.rows(ratio=1.30))
        self.assertEqual(r["decision"], "C61R_CONFIRMATORY_FAIL")
        self.assertFalse(r["nmse_ratio"]["pass"])

    def test_insufficient_eligible(self):
        r = evaluate(self.rows(n=7))
        self.assertEqual(r["decision"], "STOP_INSUFFICIENT_ELIGIBLE")

    def test_unexpected_seed_rejected(self):
        rows = self.rows(n=8)
        rows[0]["seed"] = 49400
        with self.assertRaises(ValueError):
            evaluate(rows)


if __name__ == "__main__":
    unittest.main()
