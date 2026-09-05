from __future__ import annotations

import unittest

from scripts.gaussian_shift_interface.evaluate_c66r import evaluate

SIGMAS = (0.04, 0.08, 0.12, 0.16, 0.20)


class C66REvaluatorTests(unittest.TestCase):
    def row(self, seed, gaps_pp=None, ratios=None):
        if gaps_pp is None:
            gaps_pp = [-0.2] * len(SIGMAS)
        if ratios is None:
            ratios = [1.02] * len(SIGMAS)
        conditions = {}
        for sigma, gap_pp, ratio in zip(SIGMAS, gaps_pp, ratios):
            p2_acc = 0.95
            p0_acc = p2_acc + gap_pp / 100.0
            p2_nmse = 0.10
            p0_nmse = p2_nmse * ratio
            conditions[f"{sigma:.2f}"] = {
                "sigma": sigma,
                "p0_validation_accuracy": p0_acc,
                "p2_validation_accuracy": p2_acc,
                "p0_nmse": p0_nmse,
                "p2_nmse": p2_nmse,
                "mechanism": {
                    "teacher_shifted_validation_accuracy": 0.94 - sigma / 10,
                    "teacher_accuracy_drop_from_clean_pp": -sigma * 10,
                    "frozen_hierarchy_activation_nmse_vs_teacher": 0.08 + sigma,
                    "p2_euclidean_capture_fraction": 0.2,
                    "p2_logit_l2_retained_ratio": 0.7,
                    "p2_fisher_retained_ratio": 0.8,
                },
            }
        return {
            "seed": seed,
            "eligible": True,
            "teacher_clean_validation_accuracy": 0.95,
            "sigma_conditions": conditions,
        }

    def cohort(self, gaps_pp=None, ratios=None):
        return [self.row(s, gaps_pp, ratios) for s in range(64400, 64416)]

    def test_no_failure_through_grid(self):
        out = evaluate({"rows": self.cohort()})
        self.assertEqual(out["decision"], "NO_P0_FAILURE_THROUGH_SIGMA_0_20")
        self.assertEqual(out["joint_pass_pattern"], [True] * 5)

    def test_select_first_monotone_failure(self):
        gaps = [-0.2, -0.5, -3.0, -3.5, -4.0]
        out = evaluate({"rows": self.cohort(gaps_pp=gaps)})
        self.assertEqual(out["decision"], "SELECT_SIGMA_0_12_FOR_C67R")
        self.assertEqual(out["selected_sigma_for_confirmation"], 0.12)
        self.assertEqual(out["joint_pass_pattern"], [True, True, False, False, False])

    def test_stop_on_nonmonotonic_pattern(self):
        gaps = [-0.2, -3.0, -0.2, -3.0, -3.0]
        out = evaluate({"rows": self.cohort(gaps_pp=gaps)})
        self.assertEqual(out["decision"], "STOP_NONMONOTONIC_FRONTIER")
        self.assertIsNone(out["selected_sigma_for_confirmation"])

    def test_stop_if_sigma_004_replication_unstable(self):
        gaps = [-3.0, -3.0, -3.0, -3.0, -3.0]
        out = evaluate({"rows": self.cohort(gaps_pp=gaps)})
        self.assertEqual(out["decision"], "STOP_REPLICATION_INSTABILITY_AT_SIGMA_0_04")

    def test_nmse_can_define_frontier(self):
        ratios = [1.02, 1.03, 1.40, 1.45, 1.50]
        out = evaluate({"rows": self.cohort(ratios=ratios)})
        self.assertEqual(out["decision"], "SELECT_SIGMA_0_12_FOR_C67R")

    def test_insufficient(self):
        rows = [self.row(s) for s in range(64400, 64407)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "STOP_INSUFFICIENT_ELIGIBLE")

    def test_unexpected_seed_rejected(self):
        rows = self.cohort()
        rows.append(self.row(99999))
        with self.assertRaises(ValueError):
            evaluate({"rows": rows})


if __name__ == "__main__":
    unittest.main()
