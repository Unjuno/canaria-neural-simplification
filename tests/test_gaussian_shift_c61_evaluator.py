import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "gaussian_shift_interface" / "evaluate_c61.py"
SPEC = importlib.util.spec_from_file_location("evaluate_c61", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def make_rows(count=8, val_diff=-0.005, nmse_ratio=1.02):
    rows = []
    for seed in range(49400, 49400 + count):
        p8_acc = 0.97
        p8_nmse = 0.01
        rows.append(
            {
                "seed": seed,
                "eligible": True,
                "p4_validation_accuracy": p8_acc + val_diff,
                "p8_validation_accuracy": p8_acc,
                "p4_nmse": p8_nmse * nmse_ratio,
                "p8_nmse": p8_nmse,
            }
        )
    return rows


class C61EvaluatorTests(unittest.TestCase):
    def test_stop_when_fewer_than_eight_eligible(self):
        result = MOD.evaluate(make_rows(count=7), bootstrap_seed=123)
        self.assertEqual(result["decision"], "STOP_INSUFFICIENT_ELIGIBLE")

    def test_pass_when_both_locked_margins_pass(self):
        result = MOD.evaluate(make_rows(), bootstrap_seed=123)
        self.assertEqual(result["decision"], "C61_CONFIRMATORY_PASS")
        self.assertTrue(result["validation_noninferiority"]["pass"])
        self.assertTrue(result["nmse_ratio"]["pass"])

    def test_fail_when_validation_margin_fails(self):
        result = MOD.evaluate(make_rows(val_diff=-0.03), bootstrap_seed=123)
        self.assertEqual(result["decision"], "C61_CONFIRMATORY_FAIL")
        self.assertFalse(result["validation_noninferiority"]["pass"])
        self.assertTrue(result["nmse_ratio"]["pass"])

    def test_fail_when_nmse_ratio_margin_fails(self):
        result = MOD.evaluate(make_rows(nmse_ratio=1.30), bootstrap_seed=123)
        self.assertEqual(result["decision"], "C61_CONFIRMATORY_FAIL")
        self.assertTrue(result["validation_noninferiority"]["pass"])
        self.assertFalse(result["nmse_ratio"]["pass"])

    def test_rejects_unexpected_seed(self):
        rows = make_rows()
        rows[0]["seed"] = 99999
        with self.assertRaises(ValueError):
            MOD.evaluate(rows, bootstrap_seed=123)


if __name__ == "__main__":
    unittest.main()
