from __future__ import annotations

import unittest

from scripts.gaussian_shift_interface.evaluate_c70e import evaluate


class C70EEvaluatorTests(unittest.TestCase):
    def row(
        self,
        seed,
        robust_clean=0.97,
        robust_shift=0.86,
        p32_gap_pp=-5.5,
        p64_gap_pp=-2.0,
        p32_nmse=0.10,
        p64_nmse=0.08,
        full_basis_err=1e-14,
    ):
        return {
            "seed": seed,
            "eligible": True,
            "full_basis_relative_calibration_residual_sqerr": full_basis_err,
            "robust_teacher_clean_validation_accuracy": robust_clean,
            "robust_teacher_shifted_validation_accuracy": robust_shift,
            "clean_teacher_clean_validation_accuracy": 0.98,
            "clean_teacher_shifted_validation_accuracy": 0.70,
            "p32_validation_accuracy": robust_shift + p32_gap_pp / 100.0,
            "p64_validation_accuracy": robust_shift + p64_gap_pp / 100.0,
            "p32_nmse": p32_nmse,
            "p64_nmse": p64_nmse,
        }

    def cohort(self, **kwargs):
        return [self.row(s, **kwargs) for s in range(68400, 68416)]

    def test_advance_when_p64_reference_valid(self):
        out = evaluate({"rows": self.cohort()})
        self.assertEqual(out["decision"], "ADVANCE_P64_REFERENCE_TO_C71E")
        self.assertTrue(out["robust_target_validity"]["pass"])
        self.assertTrue(out["p64_reference_validity"]["pass"])

    def test_stop_if_p64_reference_invalid(self):
        out = evaluate({"rows": self.cohort(p64_gap_pp=-7.0)})
        self.assertEqual(out["decision"], "STOP_P64_REFERENCE_INVALID")
        self.assertFalse(out["p64_reference_validity"]["pass"])

    def test_stop_if_robust_target_invalid(self):
        out = evaluate({"rows": self.cohort(robust_clean=0.97, robust_shift=0.70)})
        self.assertEqual(out["decision"], "STOP_ROBUST_TARGET_INVALID")

    def test_target_invalid_has_precedence(self):
        out = evaluate({"rows": self.cohort(robust_clean=0.97, robust_shift=0.70, p64_gap_pp=-8.0)})
        self.assertEqual(out["decision"], "STOP_ROBUST_TARGET_INVALID")

    def test_descriptive_p64_p32_does_not_gate(self):
        out = evaluate({"rows": self.cohort(p32_gap_pp=-1.0, p64_gap_pp=-2.0, p32_nmse=0.07, p64_nmse=0.08)})
        self.assertEqual(out["decision"], "ADVANCE_P64_REFERENCE_TO_C71E")
        self.assertGreater(out["p64_vs_p32_descriptive"]["nmse_geometric_mean_ratio_p64_over_p32"], 1.0)

    def test_insufficient(self):
        rows = [self.row(s) for s in range(68400, 68407)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "STOP_INSUFFICIENT_ELIGIBLE")

    def test_unexpected_seed_rejected(self):
        rows = self.cohort()
        rows.append(self.row(99999))
        with self.assertRaises(ValueError):
            evaluate({"rows": rows})


if __name__ == "__main__":
    unittest.main()
