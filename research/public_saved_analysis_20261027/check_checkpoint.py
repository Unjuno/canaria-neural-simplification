"""Compare recalculation against saved checkpoint summaries, not new evidence."""
import argparse
import json
from pathlib import Path
import analyze_saved as a

FILES = {
    'matched': ('digits_matched_regime/results/SUMMARY.json', 'digits_matched_regime/results/SUMMARY.json'),
    'grid': ('digits_joint_budget_grid/results/evidence/SUMMARY.json', 'digits_joint_budget_grid/results/evidence/SUMMARY.json'),
    'storage': ('selected_whole_storage/results/ROWS.csv', 'selected_whole_storage/results/SUMMARY.json'),
    'latency': ('selected_latency/results/RAW.jsonl', 'selected_latency/results/SUMMARY.json'),
}

EXPECTED_FIELDS = {
    "matched": ("accuracy_ci95", "accuracy_gate", "accuracy_mean_J_minus_L", "bootstrap_draws", "bootstrap_seed", "decision", "feature_gate", "nmse_sign_J_minus_L", "primary_ci95", "primary_mean_log2_J_over_L", "quality_counts"),
    "grid": ("bootstrap_draws", "bootstrap_seed", "ci95_log2", "coverage", "decision", "geometric_ratio", "mean_log2_C_over_J", "minima", "pass_counts", "patterns", "ratio_ci95"),
    "storage": ("cohort", "complete", "failures", "ratios"),
    "latency": ("paired", "ratios", "status"),
}

def check(root):
    report = {}
    for mode, (input_path, expected_path) in FILES.items():
        result = getattr(a, mode)(a.load(root / 'research' / input_path, mode))
        expected = json.loads((root / 'research' / expected_path).read_text())
        keys = sorted(EXPECTED_FIELDS[mode])
        if set(keys) - set(result):
            raise AssertionError(mode + ': missing recalculated summary fields')
        missing = set(keys) - set(expected)
        if missing:
            raise AssertionError(mode + ': missing expected summary fields: ' + repr(sorted(missing)))
        for key in keys:
            if result[key] != expected[key]:
                raise AssertionError(mode + ': exact scalar/structure mismatch: ' + key)
        report[mode] = {'exact_equal_summary_fields': keys, 'field_count': len(keys)}
        if mode == 'matched': report[mode]['recomputed_mean_nmse'] = result['mean_nmse']
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, help='root of existing public scalar checkpoint')
    args = parser.parse_args()
    print(json.dumps(check(args.checkpoint), indent=2, sort_keys=True))
