from __future__ import annotations

import unittest

from scripts.gaussian_shift_interface.evaluate_c69e import evaluate

CANDIDATES = (0, 1, 2, 4, 8, 16)


class C69EEvaluatorTests(unittest.TestCase):
    def row(
        self,
        seed,
        robust_clean=0.97,
        robust_shift=0.86,
        p32_gap_pp=-1.0,
        candidate_gaps_pp=None,
        candidate_ratios=None,
    ):
        if candidate_gaps_pp is None:
            candidate_gaps_pp = [-0.5] * len(CANDIDATES)
        if candidate_ratios is None:
            candidate_ratios = [1.05] * len(CANDIDATES)
        p32_acc = robust_shift + p32_gap_pp / 100.0
        p32_nmse = 0.08
        conditions = {
            "p32": {"validation_accuracy": p32_acc, "nmse": p32_nmse}
        }
        for k, gap, ratio in zip(CANDIDATES, candidate_gaps_pp, candidate_ratios):
            conditions[f"p{k}"] = {
                "validation_accuracy": p32_acc + gap / 100.0,
                "nmse": p32_nmse * ratio,
            }
        return {
            "seed": seed,
            "eligible": True,
            "robust_teacher_clean_validation_accuracy": robust_clean,
            "robust_teacher_shifted_validation_accuracy": robust_shift,
            "clean_teacher_clean_validation_accuracy": 0.98,
            "clean_teacher_shifted_validation_accuracy": 0.70,
            "conditions": conditions,
        }

    def cohort(self, **kwargs):
        return [self.row(s, **kwargs) for s in range(67400, 67416)]

    def test_advance_p0_when_all_candidates_pass(self):
        out = evaluate({"rows": self.cohort()})
        self.assertEqual(out["decision"], "ADVANCE_P0_TO_C70R")
        self.assertEqual(out["selected_candidate_dimension"], 0)
        self.assertEqual(out["candidate_joint_pass_pattern"], [True] * 6)

    def test_select_first_monotone_p2_candidate(self):
        gaps = [-3.0, -3.0, -0.5, -0.5, -0.5, -0.5]
        out = evaluate({"rows": self.cohort(candidate_gaps_pp=gaps)})
        self.assertEqual(out["decision"], "ADVANCE_P2_TO_C70R")
        self.assertEqual(out["selected_candidate_dimension"], 2)
        self.assertEqual(out["candidate_joint_pass_pattern"], [False, False, True, True, True, True])

    def test_nmse_gate_can_define_threshold(self):
        ratios = [1.40, 1.40, 1.05, 1.05, 1.05, 1.05]
        out = evaluate({"rows": self.cohort(candidate_ratios=ratios)})
        self.assertEqual(out["decision"], "ADVANCE_P2_TO_C70R")

    def test_stop_nonmonotonic_dimension_frontier(self):
        gaps = [-3.0, -0.5, -3.0, -0.5, -0.5, -0.5]
        out = evaluate({"rows": self.cohort(candidate_gaps_pp=gaps)})
        self.assertEqual(out["decision"], "STOP_NONMONOTONIC_DIMENSION_FRONTIER")
        self.assertIsNone(out["selected_candidate_dimension"])

    def test_no_reduced_candidate(self):
        gaps = [-3.0] * len(CANDIDATES)
        out = evaluate({"rows": self.cohort(candidate_gaps_pp=gaps)})
        self.assertEqual(out["decision"], "NO_REDUCED_CANDIDATE_THROUGH_P16")
        self.assertIsNone(out["selected_candidate_dimension"])

    def test_stop_if_robust_target_invalid(self):
        out = evaluate({"rows": self.cohort(robust_clean=0.97, robust_shift=0.70)})
        self.assertEqual(out["decision"], "STOP_ROBUST_TARGET_INVALID")

    def test_stop_if_p32_reference_invalid(self):
        out = evaluate({"rows": self.cohort(p32_gap_pp=-7.0)})
        self.assertEqual(out["decision"], "STOP_P32_REFERENCE_INVALID")

    def test_target_validity_has_precedence_over_reference(self):
        out = evaluate({"rows": self.cohort(robust_clean=0.97, robust_shift=0.70, p32_gap_pp=-7.0)})
        self.assertEqual(out["decision"], "STOP_ROBUST_TARGET_INVALID")

    def test_insufficient(self):
        rows = [self.row(s) for s in range(67400, 67407)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "STOP_INSUFFICIENT_ELIGIBLE")

    def test_unexpected_seed_rejected(self):
        rows = self.cohort()
        rows.append(self.row(99999))
        with self.assertRaises(ValueError):
            evaluate({"rows": rows})


if __name__ == "__main__":
    unittest.main()
