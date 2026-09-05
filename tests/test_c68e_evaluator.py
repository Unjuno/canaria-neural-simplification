from __future__ import annotations

import unittest

from scripts.gaussian_shift_interface.evaluate_c68e import evaluate


class C68EEvaluatorTests(unittest.TestCase):
    def row(self, seed, clean_clean=0.98, clean_shift=0.70, aug_clean=0.97, aug_shift=0.90):
        return {
            "seed": seed,
            "eligible": True,
            "clean_teacher_clean_validation_accuracy": clean_clean,
            "clean_teacher_shifted_validation_accuracy": clean_shift,
            "augmented_teacher_clean_validation_accuracy": aug_clean,
            "augmented_teacher_shifted_validation_accuracy": aug_shift,
        }

    def cohort(self, **kwargs):
        return [self.row(s, **kwargs) for s in range(66400, 66416)]

    def test_advance_if_all_repair_gates_pass(self):
        out = evaluate({"rows": self.cohort()})
        self.assertEqual(out["decision"], "ADVANCE_REPAIRED_TEACHER_TO_C69E")
        self.assertEqual(out["failed_gates"], [])
        self.assertTrue(out["clean_accuracy_noninferiority"]["pass"])
        self.assertTrue(out["shifted_accuracy_superiority"]["pass"])
        self.assertTrue(out["augmented_teacher_task_validity"]["pass"])

    def test_stop_on_clean_cost(self):
        out = evaluate({"rows": self.cohort(aug_clean=0.90, aug_shift=0.82)})
        self.assertEqual(out["decision"], "STOP_TEACHER_VALIDITY_REPAIR_GATES_FAILED")
        self.assertIn("clean_accuracy_noninferiority", out["failed_gates"])

    def test_stop_on_no_shift_gain(self):
        out = evaluate({"rows": self.cohort(clean_shift=0.90, aug_shift=0.90)})
        self.assertEqual(out["decision"], "STOP_TEACHER_VALIDITY_REPAIR_GATES_FAILED")
        self.assertIn("shifted_accuracy_superiority", out["failed_gates"])

    def test_stop_if_augmented_teacher_still_invalid(self):
        out = evaluate({"rows": self.cohort(aug_clean=0.97, aug_shift=0.70)})
        self.assertEqual(out["decision"], "STOP_TEACHER_VALIDITY_REPAIR_GATES_FAILED")
        self.assertIn("augmented_teacher_task_validity", out["failed_gates"])

    def test_multiple_failed_gates_are_reported(self):
        out = evaluate({"rows": self.cohort(clean_shift=0.75, aug_clean=0.90, aug_shift=0.70)})
        self.assertGreaterEqual(len(out["failed_gates"]), 2)

    def test_insufficient(self):
        rows = [self.row(s) for s in range(66400, 66407)]
        out = evaluate({"rows": rows})
        self.assertEqual(out["decision"], "STOP_INSUFFICIENT_ELIGIBLE")

    def test_unexpected_seed_rejected(self):
        rows = self.cohort()
        rows.append(self.row(99999))
        with self.assertRaises(ValueError):
            evaluate({"rows": rows})


if __name__ == "__main__":
    unittest.main()
