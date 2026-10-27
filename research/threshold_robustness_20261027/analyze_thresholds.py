"""Post-hoc scalar-only threshold robustness; no model/data/third-party imports."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

BUDGETS = (512, 1024, 1536, 2048, 3072, 4096, 6144)
NMSE_CEILINGS = (.06, .07, .08, .09, .10)
ERROR_ALLOWANCES = (3, 4, 5, 6, 7)
ARMS = ('C', 'J')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def integer(value, low, high, label):
    require(type(value) is int and low <= value <= high, label)

def validate(rows):
    require(type(rows) is list and len(rows) == 16, 'exactly 16 seeds required')
    require(all(type(r['id']) is int for r in rows), 'integer seed IDs required')
    require({r['id'] for r in rows} == set(range(16)), 'all unique seeds 0..15 required')
    expected = {(a, b) for a in ARMS for b in BUDGETS}
    for r in rows:
        require(r['status'] == 'COMPLETE', 'incomplete source seed')
        integer(r['teacher_correct'], 0, 270, 'invalid teacher count')
        require(len(r['cells']) == 14, '14 cells per seed required')
        require(all(type(c['budget']) is int for c in r['cells']), 'integer budgets required')
        require({(c['arm'], c['budget']) for c in r['cells']} == expected,
                'missing, duplicate, or unknown cell')
        for c in r['cells']:
            integer(c['n'], 270, 270, 'n must be 270')
            integer(c['correct'], 0, 270, 'invalid correct count')
            require(type(c['nmse']) in (int, float) and math.isfinite(c['nmse']) and c['nmse'] >= 0,
                    'invalid NMSE')
            require(type(c['reused']) is bool and c['reused'] == (c['budget'] == 2048),
                    'reuse flag mismatch')
            require(type(c['quality_pass']) is bool, 'boolean quality flag required')
            float_accuracy_pass = c['correct'] / 270 >= r['teacher_correct'] / 270 - .02
            integer_accuracy_pass = r['teacher_correct'] - c['correct'] <= 5
            require(float_accuracy_pass == integer_accuracy_pass, 'original float/count gate mismatch')
            require(c['quality_pass'] == (c['nmse'] <= .08 and float_accuracy_pass),
                    'stored original quality flag mismatch')
    return sorted(rows, key=lambda r: r['id'])

def passes(cell, teacher_correct, ceiling, allowance):
    return cell['nmse'] <= ceiling and teacher_correct - cell['correct'] <= allowance

def setting(rows, ceiling, allowance):
    minima = []
    counts = {a: {str(b): 0 for b in BUDGETS} for a in ARMS}
    coverage = dict(both=0, C_only=0, J_only=0, neither=0)
    for row in rows:
        minimum = {'id': row['id']}
        for arm in ARMS:
            successes = []
            for cell in row['cells']:
                if cell['arm'] == arm and passes(cell, row['teacher_correct'], ceiling, allowance):
                    successes.append(cell['budget'])
                    counts[arm][str(cell['budget'])] += 1
            minimum[arm] = min(successes) if successes else None
        key = ('both' if minimum['C'] is not None and minimum['J'] is not None else
               'C_only' if minimum['C'] is not None else 'J_only' if minimum['J'] is not None else 'neither')
        coverage[key] += 1
        minima.append(minimum)
    observed = coverage['both'] == 16
    direction = None
    ratio = None
    if observed:
        ratio = math.exp(math.fsum(math.log(m['C'] / m['J']) for m in minima) / 16)
        direction = {'smaller': sum(m['C'] < m['J'] for m in minima),
                     'tied': sum(m['C'] == m['J'] for m in minima),
                     'larger': sum(m['C'] > m['J'] for m in minima)}
    return dict(nmse_ceiling=ceiling, max_additional_errors=allowance,
                allowed_accuracy_drop_pp=allowance / 270 * 100,
                original_gate=(ceiling == .08 and allowance == 5),
                status='ALL_PAIRS_OBSERVED' if observed else 'CENSORED_NO_COHORT_RATIO',
                coverage=coverage, geometric_C_over_J=ratio, direction_counts=direction,
                first_passes=minima, pass_counts=counts)

def analyze(rows):
    rows = validate(rows)
    surface = [setting(rows, ceiling, allowance) for ceiling in NMSE_CEILINGS for allowance in ERROR_ALLOWANCES]
    indexed = {(s['nmse_ceiling'], s['max_additional_errors']): s for s in surface}
    original = indexed[(.08, 5)]
    require(original['coverage']['both'] == 16, 'original complete-pair gate must be open')
    margins = []
    for row, minimum in zip(rows, original['first_passes']):
        for arm in ARMS:
            cell = next(c for c in row['cells'] if c['arm'] == arm and c['budget'] == minimum[arm])
            count_slack = cell['correct'] - (row['teacher_correct'] - 5)
            tight = indexed[(.08, 4)]['first_passes'][row['id']][arm]
            loose = indexed[(.08, 6)]['first_passes'][row['id']][arm]
            margins.append(dict(id=row['id'], arm=arm, original_budget=minimum[arm],
                                teacher_correct=row['teacher_correct'], correct=cell['correct'],
                                nmse=cell['nmse'], additional_errors=row['teacher_correct'] - cell['correct'],
                                correct_count_slack=count_slack, correct_losses_to_fail=count_slack + 1,
                                nmse_slack=.08 - cell['nmse'],
                                tightened_allow4_budget=tight, tightened_selection_changed=tight != minimum[arm],
                                loosened_allow6_budget=loose, loosened_selection_changed=loose != minimum[arm]))
    return dict(analysis_class='POST_HOC_DESCRIPTIVE_SAVED_VALIDATION_SCALARS',
                original_float_integer_equivalence_checked_cells=224,
                settings=surface, selected_margins=margins,
                no_new_confidence_intervals=True, no_new_data_or_model_execution=True)

def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw)
    result = analyze(data['rows'])
    original = next(s for s in result['settings'] if s['original_gate'])
    require(abs(original['geometric_C_over_J'] - data['accepted_original']['geometric_ratio']) < 1e-14,
            'accepted original geometric ratio mismatch')
    require(original['first_passes'] == data['accepted_original']['minima'], 'accepted original minima mismatch')
    require(original['direction_counts'] == dict(smaller=11, tied=4, larger=1), 'accepted direction counts mismatch')
    result['input_scalar_sha256'] = hashlib.sha256(raw).hexdigest()
    result['accepted_source_commit'] = data['accepted_source_commit']
    result['accepted_source_summary_sha256'] = data['accepted_source_summary_sha256']
    args.output.mkdir(exist_ok=False)
    (args.output / 'RESULTS.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    surface_rows = []
    minima_rows = []
    pass_rows = []
    for s in result['settings']:
        surface_rows.append(dict(nmse_ceiling=s['nmse_ceiling'], max_additional_errors=s['max_additional_errors'],
                                 allowed_accuracy_drop_pp=s['allowed_accuracy_drop_pp'], original_gate=s['original_gate'],
                                 status=s['status'], **s['coverage'], geometric_C_over_J=s['geometric_C_over_J'],
                                 **(s['direction_counts'] or dict(smaller=None, tied=None, larger=None))))
        for m in s['first_passes']:
            minima_rows.append(dict(nmse_ceiling=s['nmse_ceiling'], max_additional_errors=s['max_additional_errors'], **m))
        for b in BUDGETS:
            pass_rows.append(dict(nmse_ceiling=s['nmse_ceiling'], max_additional_errors=s['max_additional_errors'],
                                  budget=b, C=s['pass_counts']['C'][str(b)], J=s['pass_counts']['J'][str(b)]))
    write_csv(args.output / 'SURFACE.csv', surface_rows)
    write_csv(args.output / 'FIRST_PASSES.csv', minima_rows)
    write_csv(args.output / 'PASS_COUNTS.csv', pass_rows)
    write_csv(args.output / 'SELECTED_MARGINS.csv', result['selected_margins'])
    print(json.dumps({'original': original, 'surface': surface_rows, 'margins': result['selected_margins']}, indent=2))

if __name__ == '__main__':
    main()
