from __future__ import annotations

import unittest

from scripts.gaussian_shift_interface.evaluate_c71e import evaluate


class C71EEvaluatorTests(unittest.TestCase):
    def row(
        self,
        seed,
        robust_clean=0.97,
        robust_shift=0.86,
        h_gap_pp=-2.0,
        s_gap_pp=-2.0,
        d_gap_pp=-2.0,
    ):
        return {
            "seed": seed,
            "eligible": True,
            "full_basis_relative_calibration_residual_sqerr": 1e-13,
            "robust_teacher_clean_validation_accuracy": robust_clean,
            "robust_teacher_shifted_validation_accuracy": robust_shift,
            "clean_teacher_clean_validation_accuracy": 0.98,
            "clean_teacher_shifted_validation_accuracy": 0.70,
            "h64_validation_accuracy": robust_shift + h_gap_pp / 100.0,
            "s64_validation_accuracy": robust_shift + s_gap_pp / 100.0,
            "d64_validation_accuracy": robust_shift + d_gap_pp / 100.0,
            "h64_nmse_vs_teacher": 0.08,
            "s64_nmse_vs_teacher": 0.09,
            "d64_nmse_vs_teacher": 0.07,
            "s64_nmse_vs_h64": 0.03,
            "h64_calibration_nmse_vs_teacher": 0.01,
            "d64_calibration_nmse_vs_teacher": 0.02,
        }

    def cohort(self, **kwargs):
        return [self.row(s, **kwargs) for s in range(69400, 69416)]

    def test_not_reproduced_if_h_and_s_valid(self):
        out = evaluate({"rows": self.cohort(h_gap_pp=-2, s_gap_pp=-2, d_gap_pp=-2)})
        self.assertEqual(out["decision"], "STOP_C70_P64_FAILURE_NOT_REPRODUCED")
        self.assertEqual(out["stage_validity_pattern"], [True, True, True])

    def test_nonmonotonic_compiler_rescue(self):
        out = evaluate({"rows": self.cohort(h_gap_pp=-7, s_gap_pp=-2, d_gap_pp=-2)})
        self.assertEqual(out["decision"], "STOP_NONMONOTONIC_COMPILER_RESCUE")
        self.assertEqual(out["stage_validity_pattern"], [False, True, True])

    def test_localize_standard_compilation_loss(self):
        out = evaluate({"rows": self.cohort(h_gap_pp=-2, s_gap_pp=-7, d_gap_pp=-2)})
        self.assertEqual(out["decision"], "LOCALIZE_STANDARD_COMPILATION_LOSS")
        self.assertEqual(out["stage_validity_pattern"], [True, False, True])

    def test_localize_hierarchy_adaptation_loss_if_direct_valid(self):
        out = evaluate({"rows": self.cohort(h_gap_pp=-7, s_gap_pp=-7, d_gap_pp=-2)})
        self.assertEqual(out["decision"], "LOCALIZE_HIERARCHY_ADAPTATION_LOSS")
        self.assertEqual(out["stage_validity_pattern"], [False, False, True])

    def test_localize_shared_mapping_or_calibration_limit(self):
        out = evaluate({"rows": self.cohort(h_gap_pp=-7, s_gap_pp=-7, d_gap_pp=-7)})
        self.assertEqual(out["decision"], "LOCALIZE_SHARED_MAPPING_OR_CALIBRATION_LIMIT")
        self.assertEqual(out["stage_validity_pattern"], [False, False, False])

    def test_h_valid_s_invalid_has_compilation_precedence_even_if_d_invalid(self):
        out = evaluate({"rows": self.cohort(h_gap_pp=-2, s_gap_pp=-7, d_gap_pp=-7)})
        self.assertEqual(out["decision"], "LOCALIZE_STANDARD_COMPILATION_LOSS")

    def test_target_invalid_has_precedence(self):
        out = evaluate({"rows": self.cohort(robust_clean=0.97, robust_shift=0.70, h_gap_pp=-7, s_gap_pp=-7, d_gap_pp=-2)})
        self.assertEqual(out["decision"], "STOP_ROBUST_TARGET_INVALID")

    def test_insufficient(self):
        rows = [self.row(s) for s in range(69400, 69407)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "STOP_INSUFFICIENT_ELIGIBLE")

    def test_unexpected_seed_rejected(self):
        rows = self.cohort()
        rows.append(self.row(99999))
        with self.assertRaises(ValueError):
            evaluate({"rows": rows})


if __name__ == "__main__":
    unittest.main()
