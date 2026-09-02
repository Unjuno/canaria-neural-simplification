from __future__ import annotations

import unittest

from scripts.gaussian_shift_interface.evaluate_c64r import evaluate


class C64REvaluatorTests(unittest.TestCase):
    def row(self, seed, p0_acc, p1_acc, p2_acc, p0_nmse, p1_nmse, p2_nmse):
        geom = {
            "shifted_calibration": {},
            "shifted_validation": {},
        }
        for split in geom:
            geom[split] = {
                "p0": {
                    "dimension": 0,
                    "euclidean_capture_fraction": 0.0,
                    "logit_l2_retained_ratio": 1.0,
                    "fisher_retained_ratio": 1.0,
                },
                "p1": {
                    "dimension": 1,
                    "euclidean_capture_fraction": 0.15,
                    "logit_l2_retained_ratio": 0.78,
                    "fisher_retained_ratio": 0.72,
                },
                "p2": {
                    "dimension": 2,
                    "euclidean_capture_fraction": 0.25,
                    "logit_l2_retained_ratio": 0.62,
                    "fisher_retained_ratio": 0.55,
                },
            }
        return {
            "seed": seed,
            "eligible": True,
            "conditions": {
                "p0": {"validation_accuracy": p0_acc, "nmse": p0_nmse},
                "p1": {"validation_accuracy": p1_acc, "nmse": p1_nmse},
                "p2": {"validation_accuracy": p2_acc, "nmse": p2_nmse},
            },
            "diagnostics": {"task_weighted_geometry": geom},
        }

    def test_advance_p0(self):
        rows = [self.row(s, .95, .951, .952, .10, .099, .098) for s in range(62400, 62416)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "ADVANCE_P0_TO_C65R")
        self.assertTrue(out["primary_exploratory_frontier"]["p0_vs_p2"]["joint_pass"])

    def test_advance_p1_when_p0_fails(self):
        rows = [self.row(s, .90, .951, .952, .16, .099, .098) for s in range(62400, 62416)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "ADVANCE_P1_TO_C65R")
        self.assertFalse(out["primary_exploratory_frontier"]["p0_vs_p2"]["joint_pass"])
        self.assertTrue(out["primary_exploratory_frontier"]["p1_vs_p2"]["joint_pass"])

    def test_stop_at_p2(self):
        rows = [self.row(s, .90, .91, .95, .16, .15, .10) for s in range(62400, 62416)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "STOP_LOWER_FRONTIER_AT_P2")

    def test_insufficient(self):
        rows = [self.row(s, .95, .951, .952, .10, .099, .098) for s in range(62400, 62407)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "STOP_INSUFFICIENT_ELIGIBLE")


if __name__ == "__main__":
    unittest.main()
