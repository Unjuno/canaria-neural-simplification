from __future__ import annotations

import unittest

from scripts.gaussian_shift_interface.evaluate_c72e import evaluate

CELLS = ("N192_W32", "N384_W32", "N192_W64", "N384_W64")


class C72EEvaluatorTests(unittest.TestCase):
    def row(self, seed, gaps_pp=None, robust_clean=0.97, robust_shift=0.86):
        if gaps_pp is None:
            gaps_pp = [-7.0, -3.0, -7.0, -3.0]
        cells = {}
        for c, gap in zip(CELLS, gaps_pp):
            cells[c] = {
                "validation_accuracy": robust_shift + gap / 100.0,
                "activation_nmse_vs_teacher": 0.08,
                "calibration_nmse_vs_teacher": 0.01,
                "calibration_samples": 192 if "N192" in c else 384,
                "trainable_parameters": 4096 if "W32" in c else 8192,
            }
        return {
            "seed": seed,
            "eligible": True,
            "robust_teacher_clean_validation_accuracy": robust_clean,
            "robust_teacher_shifted_validation_accuracy": robust_shift,
            "clean_teacher_clean_validation_accuracy": 0.98,
            "clean_teacher_shifted_validation_accuracy": 0.70,
            "cells": cells,
        }

    def cohort(self, **kwargs):
        return [self.row(s, **kwargs) for s in range(70400, 70416)]

    def test_calibration_quantity_repair(self):
        out = evaluate({"rows": self.cohort(gaps_pp=[-7,-3,-7,-3])})
        self.assertEqual(out["decision"], "LOCALIZE_CALIBRATION_QUANTITY_REPAIR")
        self.assertEqual(out["cell_validity_pattern"], [False, True, False, True])

    def test_capacity_repair(self):
        out = evaluate({"rows": self.cohort(gaps_pp=[-7,-7,-3,-3])})
        self.assertEqual(out["decision"], "LOCALIZE_MAPPING_CAPACITY_REPAIR")
        self.assertEqual(out["cell_validity_pattern"], [False, False, True, True])

    def test_both_single_factor_repairs(self):
        out = evaluate({"rows": self.cohort(gaps_pp=[-7,-3,-3,-2])})
        self.assertEqual(out["decision"], "BOTH_SINGLE_FACTOR_REPAIRS")
        self.assertEqual(out["cell_validity_pattern"], [False, True, True, True])

    def test_interaction_repair(self):
        out = evaluate({"rows": self.cohort(gaps_pp=[-7,-7,-7,-3])})
        self.assertEqual(out["decision"], "LOCALIZE_CALIBRATION_CAPACITY_INTERACTION_REPAIR")
        self.assertEqual(out["cell_validity_pattern"], [False, False, False, True])

    def test_no_repair(self):
        out = evaluate({"rows": self.cohort(gaps_pp=[-7,-7,-7,-7])})
        self.assertEqual(out["decision"], "NO_REPAIR_AT_N384_W64")

    def test_nonmonotonic_factorial_repair(self):
        out = evaluate({"rows": self.cohort(gaps_pp=[-7,-3,-7,-7])})
        self.assertEqual(out["decision"], "STOP_NONMONOTONIC_FACTORIAL_REPAIR")

    def test_baseline_reproduction_precedence(self):
        out = evaluate({"rows": self.cohort(gaps_pp=[-3,-3,-3,-3])})
        self.assertEqual(out["decision"], "STOP_C71_BASELINE_FAILURE_NOT_REPRODUCED")

    def test_target_invalid_precedence(self):
        out = evaluate({"rows": self.cohort(gaps_pp=[-3,-3,-3,-3], robust_clean=0.97, robust_shift=0.70)})
        self.assertEqual(out["decision"], "STOP_ROBUST_TARGET_INVALID")

    def test_insufficient(self):
        rows = [self.row(s) for s in range(70400, 70407)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "STOP_INSUFFICIENT_ELIGIBLE")

    def test_unexpected_seed_rejected(self):
        rows = self.cohort()
        rows.append(self.row(99999))
        with self.assertRaises(ValueError):
            evaluate({"rows": rows})


if __name__ == "__main__":
    unittest.main()
