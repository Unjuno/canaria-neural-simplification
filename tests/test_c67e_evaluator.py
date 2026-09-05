from __future__ import annotations

import unittest

from scripts.gaussian_shift_interface.evaluate_c67e import evaluate

SIGMAS = (0.20, 0.28, 0.36, 0.44, 0.52, 0.60)


class C67EEvaluatorTests(unittest.TestCase):
    def row(self, seed, gaps_pp=None, ratios=None, teacher_drops_pp=None, p2_teacher_gaps_pp=None):
        if gaps_pp is None:
            gaps_pp = [-0.2] * len(SIGMAS)
        if ratios is None:
            ratios = [1.03] * len(SIGMAS)
        if teacher_drops_pp is None:
            teacher_drops_pp = [-5.0] * len(SIGMAS)
        if p2_teacher_gaps_pp is None:
            p2_teacher_gaps_pp = [-1.0] * len(SIGMAS)
        clean = 0.95
        conditions = {}
        for sigma, p0_gap, ratio, teacher_drop, ref_gap in zip(
            SIGMAS, gaps_pp, ratios, teacher_drops_pp, p2_teacher_gaps_pp
        ):
            teacher_shift = clean + teacher_drop / 100.0
            p2_acc = teacher_shift + ref_gap / 100.0
            p0_acc = p2_acc + p0_gap / 100.0
            p2_nmse = 0.10
            p0_nmse = p2_nmse * ratio
            conditions[f"{sigma:.2f}"] = {
                "sigma": sigma,
                "p0_validation_accuracy": p0_acc,
                "p2_validation_accuracy": p2_acc,
                "p0_nmse": p0_nmse,
                "p2_nmse": p2_nmse,
                "mechanism": {
                    "teacher_shifted_validation_accuracy": teacher_shift,
                    "teacher_accuracy_drop_from_clean_pp": teacher_drop,
                    "p2_accuracy_gap_to_shifted_teacher_pp": ref_gap,
                    "frozen_hierarchy_activation_nmse_vs_teacher": 0.1 + sigma,
                    "p2_euclidean_capture_fraction": 0.2,
                    "p2_logit_l2_retained_ratio": 0.7,
                    "p2_fisher_retained_ratio": 0.8,
                },
            }
        return {
            "seed": seed,
            "eligible": True,
            "teacher_clean_validation_accuracy": clean,
            "sigma_conditions": conditions,
        }

    def cohort(self, **kwargs):
        return [self.row(s, **kwargs) for s in range(65400, 65416)]

    def test_no_failure_through_grid(self):
        out = evaluate({"rows": self.cohort()})
        self.assertEqual(out["decision"], "NO_P0_FAILURE_THROUGH_SIGMA_0_60")
        self.assertEqual(out["p0_joint_pass_pattern"], [True] * 6)
        self.assertEqual(out["validity_pass_pattern"], [True] * 6)
        self.assertIsNone(out["selected_sigma_for_confirmation"])

    def test_select_first_monotone_p0_failure(self):
        gaps = [-0.2, -0.5, -3.0, -3.5, -4.0, -4.5]
        out = evaluate({"rows": self.cohort(gaps_pp=gaps)})
        self.assertEqual(out["decision"], "SELECT_SIGMA_0_36_FOR_C68R")
        self.assertEqual(out["selected_sigma_for_confirmation"], 0.36)

    def test_nmse_can_select_frontier(self):
        ratios = [1.02, 1.05, 1.40, 1.45, 1.50, 1.55]
        out = evaluate({"rows": self.cohort(ratios=ratios)})
        self.assertEqual(out["decision"], "SELECT_SIGMA_0_36_FOR_C68R")

    def test_stop_at_teacher_validity_boundary_before_p0_failure(self):
        drops = [-5, -10, -15, -25, -30, -35]
        out = evaluate({"rows": self.cohort(teacher_drops_pp=drops)})
        self.assertEqual(out["decision"], "STOP_VALIDITY_BOUNDARY_AT_SIGMA_0_44")
        self.assertEqual(out["validity_boundary_sigma"], 0.44)
        self.assertIsNone(out["selected_sigma_for_confirmation"])

    def test_stop_at_reference_validity_boundary(self):
        ref = [-1, -2, -3, -7, -8, -9]
        out = evaluate({"rows": self.cohort(p2_teacher_gaps_pp=ref)})
        self.assertEqual(out["decision"], "STOP_VALIDITY_BOUNDARY_AT_SIGMA_0_44")
        self.assertEqual(out["validity_boundary_sigma"], 0.44)

    def test_p0_failure_before_later_validity_boundary_can_select(self):
        gaps = [-0.2, -0.5, -3.0, -3.5, -4.0, -4.5]
        drops = [-5, -8, -10, -12, -25, -30]
        out = evaluate({"rows": self.cohort(gaps_pp=gaps, teacher_drops_pp=drops)})
        self.assertEqual(out["decision"], "SELECT_SIGMA_0_36_FOR_C68R")
        self.assertEqual(out["validity_boundary_sigma"], 0.52)

    def test_stop_on_nonmonotonic_p0_frontier(self):
        gaps = [-0.2, -3.0, -0.2, -3.0, -3.0, -3.0]
        out = evaluate({"rows": self.cohort(gaps_pp=gaps)})
        self.assertEqual(out["decision"], "STOP_NONMONOTONIC_P0_FRONTIER")
        self.assertIsNone(out["selected_sigma_for_confirmation"])

    def test_stop_if_anchor_p0_is_unstable(self):
        gaps = [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0]
        out = evaluate({"rows": self.cohort(gaps_pp=gaps)})
        self.assertEqual(out["decision"], "STOP_C66R_ANCHOR_P0_INSTABILITY")

    def test_stop_if_anchor_validity_is_unstable(self):
        drops = [-25, -25, -25, -25, -25, -25]
        out = evaluate({"rows": self.cohort(teacher_drops_pp=drops)})
        self.assertEqual(out["decision"], "STOP_C66R_ANCHOR_VALIDITY_INSTABILITY")
        self.assertEqual(out["validity_boundary_sigma"], 0.20)

    def test_insufficient(self):
        rows = [self.row(s) for s in range(65400, 65407)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "STOP_INSUFFICIENT_ELIGIBLE")

    def test_unexpected_seed_rejected(self):
        rows = self.cohort()
        rows.append(self.row(99999))
        with self.assertRaises(ValueError):
            evaluate({"rows": rows})


if __name__ == "__main__":
    unittest.main()
