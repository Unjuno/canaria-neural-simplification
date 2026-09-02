import unittest

from scripts.gaussian_shift_interface.evaluate_c63r import evaluate


def row(seed, p2_acc=0.950, p4_acc=0.955, p2_nmse=0.100, p4_nmse=0.095):
    return {
        'seed': seed,
        'eligible': True,
        'p2_validation_accuracy': p2_acc,
        'p4_validation_accuracy': p4_acc,
        'p2_nmse': p2_nmse,
        'p4_nmse': p4_nmse,
        'frozen_nmse': 0.120,
    }


class C63REvaluatorTests(unittest.TestCase):
    def test_confirmatory_pass(self):
        rows = [row(s) for s in range(61400, 61408)]
        out = evaluate(rows)
        self.assertEqual(out['decision'], 'C63R_CONFIRMATORY_PASS')
        self.assertTrue(out['validation_noninferiority']['pass'])
        self.assertTrue(out['nmse_ratio']['pass'])

    def test_confirmatory_fail(self):
        rows = [row(s, p2_acc=0.920, p4_acc=0.955, p2_nmse=0.140, p4_nmse=0.095) for s in range(61400, 61408)]
        out = evaluate(rows)
        self.assertEqual(out['decision'], 'C63R_CONFIRMATORY_FAIL')

    def test_insufficient_eligible(self):
        rows = [row(s) for s in range(61400, 61407)]
        out = evaluate(rows)
        self.assertEqual(out['decision'], 'STOP_INSUFFICIENT_ELIGIBLE')

    def test_unexpected_seed_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([row(99999)])


if __name__ == '__main__':
    unittest.main()
