import unittest

from scripts.gaussian_shift_interface.evaluate_c62r import evaluate


def make_row(seed: int, qr2_acc: float, qr4_acc: float, qr2_nmse: float, qr4_nmse: float):
    conditions = {
        'qr_p2': {'validation_accuracy': qr2_acc, 'nmse': qr2_nmse},
        'qr_p4': {'validation_accuracy': qr4_acc, 'nmse': qr4_nmse},
        'svd_p2': {'validation_accuracy': qr2_acc + 0.002, 'nmse': qr2_nmse * 0.98},
        'svd_p4': {'validation_accuracy': qr4_acc + 0.001, 'nmse': qr4_nmse * 0.99},
        'random_p2': {'validation_accuracy': qr2_acc - 0.01, 'nmse': qr2_nmse * 1.10},
        'random_p4': {'validation_accuracy': qr4_acc - 0.005, 'nmse': qr4_nmse * 1.05},
    }
    diagnostics = {
        'spectrum': {
            'entropy_effective_rank': 5.0 + 0.01 * (seed - 60400),
            'stable_rank': 3.0,
            'optimal_svd_energy_fraction_k2': 0.50,
            'optimal_svd_energy_fraction_k4': 0.70,
            'optimal_svd_energy_fraction_k8': 0.85,
        },
        'energy_capture': {
            'qr': {
                'k2': {'calibration': 0.30, 'shifted_validation': 0.28},
                'k4': {'calibration': 0.48, 'shifted_validation': 0.45},
            },
            'svd': {
                'k2': {'calibration': 0.50, 'shifted_validation': 0.43},
                'k4': {'calibration': 0.70, 'shifted_validation': 0.61},
            },
            'random': {
                'k2': {'calibration': 0.04, 'shifted_validation': 0.04},
                'k4': {'calibration': 0.08, 'shifted_validation': 0.08},
            },
        },
        'qr_to_svd_alignment': {
            'k2': {'mean_squared_principal_cosine': 0.35},
            'k4': {'mean_squared_principal_cosine': 0.42},
        },
    }
    return {
        'seed': seed,
        'eligible': True,
        'frozen_nmse': 0.12,
        'conditions': conditions,
        'diagnostics': diagnostics,
    }


class C62REvaluatorTests(unittest.TestCase):
    def test_advance(self):
        rows = [make_row(s, 0.950, 0.955, 0.100, 0.095) for s in range(60400, 60408)]
        out = evaluate({'rows': rows})
        self.assertEqual(out['decision'], 'ADVANCE_QR_P2_TO_C63R')
        self.assertFalse(out['confirmatory_claim_allowed'])
        self.assertTrue(out['primary_exploratory_frontier']['validation_noninferiority_pass'])
        self.assertTrue(out['primary_exploratory_frontier']['nmse_ratio_pass'])

    def test_frontier_stop(self):
        rows = [make_row(s, 0.920, 0.955, 0.130, 0.095) for s in range(60400, 60408)]
        out = evaluate({'rows': rows})
        self.assertEqual(out['decision'], 'STOP_P2_FRONTIER_AT_C62R')

    def test_insufficient(self):
        rows = [make_row(s, 0.950, 0.955, 0.100, 0.095) for s in range(60400, 60407)]
        out = evaluate({'rows': rows})
        self.assertEqual(out['decision'], 'STOP_INSUFFICIENT_ELIGIBLE')


if __name__ == '__main__':
    unittest.main()
