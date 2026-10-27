"""Portable saved-scalar analysis; NOT the original experiment runner.

Reads existing JSON/CSV/JSONL tables only. Never loads models or datasets,
executes inference, trains, times code, or accesses the network.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import numpy as np

BUDGETS = (512, 1024, 1536, 2048, 3072, 4096, 6144)
PAIRS = (('C', 'J'), ('C', 'teacher'), ('J', 'teacher'))

class InvalidInput(ValueError):
    """The fixed-cohort scalar input is incomplete or inconsistent."""

def require(ok, message):
    if not ok:
        raise InvalidInput(message)

def number(x, name, lower=0):
    require(not isinstance(x, bool) and isinstance(x, (int, float)), name + ': numeric required')
    require(math.isfinite(x) and x >= lower, name + ': finite value above lower bound required')
    return x

def integer(x, name, lower=0, upper=None):
    require(type(x) is int and x >= lower and (upper is None or x <= upper), name + ': invalid integer')
    return x

def cohort(rows):
    require(isinstance(rows, list) and len(rows) == 16, 'exactly all 16 rows required; no exclusions')
    ids = [integer(r['id'], 'id', 0, 15) for r in rows]
    require(set(ids) == set(range(16)), 'duplicate or missing seed id')
    return sorted(rows, key=lambda r: r['id'])

def quality(cell, teacher_correct):
    integer(teacher_correct, 'teacher_correct', 0, 270)
    require(cell['n'] == 270 and type(cell['n']) is int, 'n must be 270')
    correct = integer(cell['correct'], 'correct', 0, 270)
    nmse = number(cell['nmse'], 'nmse')
    actual = nmse <= .08 and correct / 270 >= teacher_correct / 270 - .02
    require(type(cell['quality_pass']) is bool and cell['quality_pass'] == actual, 'quality flag disagrees with fixed thresholds')
    if 'accuracy' in cell:
        number(cell['accuracy'], 'accuracy')
        require(cell['accuracy'] == correct / 270, 'accuracy/count mismatch')
    return actual

def bootstrap(values, seed):
    indices = np.random.default_rng(seed).integers(0, 16, (100000, 16))
    return np.quantile(np.asarray(values)[indices].mean(1), [.025, .975])

def matched(rows):
    rows = cohort(rows)
    for row in rows:
        require(row['status'] == 'COMPLETE', 'incomplete seed')
        require(set(row['arms']) == {'C', 'J', 'L'}, 'exactly C/J/L required')
        for cell in row['arms'].values():
            quality(cell, row['teacher_correct'])
    nm = np.array([math.log2(max(r['arms']['J']['nmse'], 1e-12) / max(r['arms']['L']['nmse'], 1e-12)) for r in rows])
    acc = np.array([(r['arms']['J']['correct'] - r['arms']['L']['correct']) / 270 for r in rows])
    seed = 271027042
    ci, ai = bootstrap(nm, seed), bootstrap(acc, seed)
    fg = 'PASS' if ci[1] < 0 else 'FAIL' if ci[0] > 0 else 'UNCERTAIN'
    ag = 'PASS' if ai[0] > -.02 else 'FAIL' if ai[1] < -.02 else 'UNCERTAIN'
    return dict(decision='CONDITIONAL_REGIME_SUPPORTED' if fg == ag == 'PASS' else 'NOT_JOINTLY_SUPPORTED',
                feature_gate=fg, accuracy_gate=ag, primary_mean_log2_J_over_L=float(nm.mean()),
                primary_ci95=ci.tolist(), accuracy_mean_J_minus_L=float(acc.mean()), accuracy_ci95=ai.tolist(),
                bootstrap_seed=seed, bootstrap_draws=100000,
                mean_nmse={a: float(np.mean([r['arms'][a]['nmse'] for r in rows])) for a in ('L', 'J', 'C')},
                quality_counts={a: sum(quality(r['arms'][a], r['teacher_correct']) for r in rows) for a in ('L', 'J', 'C')},
                nmse_sign_J_minus_L={'lower': int((nm < 0).sum()), 'equal': int((nm == 0).sum()), 'higher': int((nm > 0).sum())})

def grid(rows):
    rows = cohort(rows)
    minima, patterns = [], []
    counts = {a: {str(b): 0 for b in BUDGETS} for a in ('C', 'J')}
    for row in rows:
        require(row['status'] == 'COMPLETE', 'incomplete seed')
        cells = row['cells']
        require(len(cells) == 14, '14 cells required')
        for c in cells:
            integer(c['budget'], 'budget')
        require({(c['arm'], c['budget']) for c in cells} == {(a, b) for a in ('C', 'J') for b in BUDGETS}, 'duplicate/missing grid cell')
        for c in cells:
            require(type(c['reused']) is bool and c['reused'] == (c['budget'] == 2048), 'reused cell mismatch')
            counts[c['arm']][str(c['budget'])] += int(quality(c, row['teacher_correct']))
        m = {'id': row['id']}
        for arm in ('C', 'J'):
            by = {c['budget']: c['quality_pass'] for c in cells if c['arm'] == arm}
            passing = [b for b in BUDGETS if by[b]]
            m[arm] = min(passing) if passing else None
            patterns.append(dict(id=row['id'], arm=arm, pass_by_budget=[by[b] for b in BUDGETS],
                                 nonmonotone=any(by[b] and not by[d] for b in BUDGETS for d in BUDGETS if d > b)))
        minima.append(m)
    coverage = {'both': 0, 'C_only': 0, 'J_only': 0, 'neither': 0}
    for m in minima:
        key = 'both' if m['C'] and m['J'] else 'C_only' if m['C'] else 'J_only' if m['J'] else 'neither'
        coverage[key] += 1
    out = dict(decision='CENSORED_NO_COHORT_RATIO', minima=minima, patterns=patterns, coverage=coverage, pass_counts=counts)
    if coverage['both'] == 16:
        v = np.array([math.log2(m['C'] / m['J']) for m in minima])
        ci = bootstrap(v, 2710276144)
        out.update(decision='CONDITIONAL_GRID_ADVANTAGE' if ci[1] < 0 else 'CONDITIONAL_GRID_DISADVANTAGE' if ci[0] > 0 else 'UNCERTAIN',
                   mean_log2_C_over_J=float(v.mean()), ci95_log2=ci.tolist(), geometric_ratio=float(2**v.mean()),
                   ratio_ci95=(2**ci).tolist(), bootstrap_seed=2710276144, bootstrap_draws=100000)
    return out

def ratio_summary(values, storage=False):
    logs = list(map(math.log, values))
    avg = sum(logs) / len(logs) if storage else statistics.mean(logs)
    return dict(geomean=math.exp(avg), min=min(values), max=max(values),
                smaller=sum(v < 1 for v in values), tie=sum(v == 1 for v in values), larger=sum(v > 1 for v in values))

def storage(rows):
    rows = cohort(rows)
    ratios = {}
    for metric in ('parameters', 'stored_bytes', 'deflated_bytes'):
        for a, b in PAIRS:
            key = a + '_over_' + b + '_' + metric
            values = []
            for row in rows:
                numerator = integer(row[a + '_' + metric], a + '_' + metric, 1)
                denominator = integer(row[b + '_' + metric], b + '_' + metric, 1)
                values.append(numerator / denominator)
            ratios[key] = ratio_summary(values, storage=True)
    return dict(cohort=16, complete=16, failures=[], ratios=ratios)

def latency(raw):
    require(len(raw) == 2880, 'all 2880 timing blocks required')
    grouped, identities, positions = {}, set(), set()
    for r in raw:
        sid = integer(r['id'], 'id', 0, 15)
        batch = integer(r['batch'], 'batch')
        require(batch in (1, 128) and r['arm'] in ('teacher', 'C', 'J'), 'invalid batch/arm')
        rnd = integer(r['round'], 'round', 0, 29)
        pos = integer(r['position'], 'position', 0, 2)
        require(type(r['iterations']) is int and r['iterations'] == 20, '20 forwards/block required')
        total = integer(r['total_ns'], 'total_ns', 1)
        per = number(r['ns_per_forward'], 'ns_per_forward', 1e-100)
        require(per == total / 20, 'timing arithmetic mismatch')
        identity = sid, batch, rnd, r['arm']
        position = sid, batch, rnd, pos
        require(identity not in identities and position not in positions, 'duplicate timing identity/position')
        identities.add(identity); positions.add(position)
        grouped.setdefault((sid, batch, r['arm']), []).append(per)
    medians = []
    for sid in range(16):
        for batch in (1, 128):
            row = {'id': sid, 'batch': batch}
            for arm in ('teacher', 'C', 'J'):
                values = grouped.get((sid, batch, arm), [])
                require(len(values) == 30, '30 blocks per cell required')
                row[arm + '_ns'] = statistics.median(values)
            for a, b in PAIRS:
                row[a + '_over_' + b] = row[a + '_ns'] / row[b + '_ns']
            medians.append(row)
    ratios = {}
    for batch in (1, 128):
        for a, b in PAIRS:
            key = a + '_over_' + b
            ratios[f'batch{batch}_{key}'] = ratio_summary([r[key] for r in medians if r['batch'] == batch])
    return dict(status='COMPLETE', paired=medians, ratios=ratios)

def load(path, mode):
    path = Path(path)
    if mode == 'latency':
        with path.open() as f:
            return [json.loads(line) for line in f if line.strip()]
    if mode == 'storage':
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        # Ignore derived ratio columns; recompute from the recorded numerators.
        fields = ['id'] + [a + '_' + m for a in ('C', 'J', 'teacher') for m in ('parameters', 'stored_bytes', 'deflated_bytes')]
        return [{k: int(r[k]) for k in fields} for r in rows]
    with path.open() as f:
        doc = json.load(f)
    return doc['all_rows']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('matched', 'grid', 'storage', 'latency'))
    parser.add_argument('input', help='saved scalar JSON, CSV (storage), or JSONL (latency)')
    args = parser.parse_args()
    try:
        result = globals()[args.mode](load(args.input, args.mode))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.exit(2, 'Invalid saved input: ' + str(exc) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))

if __name__ == '__main__':
    main()
