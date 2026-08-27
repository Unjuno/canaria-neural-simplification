from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = list(range(1380, 1388))
BOOTSTRAP_SEED = 20260829
N_BOOT = 100_000


def pct95(x):
    q = np.quantile(x, [0.025, 0.975])
    return [float(q[0]), float(q[1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, default=Path('results/confirmatory/c7_depth2_recursive/seed_rows.csv'))
    args = ap.parse_args()
    with args.csv.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: int(r['seed']))
    assert [int(r['seed']) for r in rows] == EXPECTED_SEEDS

    hf = np.array([float(r['hf_nmse']) for r in rows])
    hj = np.array([float(r['hj_nmse']) for r in rows])
    sl = np.array([float(r['sl_nmse']) for r in rows])
    direct = np.array([float(r['direct_nmse']) for r in rows])
    hjv = np.array([float(r['hj_val_acc']) for r in rows])
    dv = np.array([float(r['direct_val_acc']) for r in rows])
    hjt = np.array([float(r['hj_test_acc']) for r in rows])
    dt = np.array([float(r['direct_test_acc']) for r in rows])

    d_frozen = hj - hf
    r_joint = hj / direct
    d_single = hj - sl
    r_strict = hf / direct
    val_diff = hjv - dv
    test_diff = hjt - dt

    stored = {
        'D_frozen': np.array([float(r['D_frozen']) for r in rows]),
        'R_joint': np.array([float(r['R_joint']) for r in rows]),
        'D_single': np.array([float(r['D_single']) for r in rows]),
        'R_strict': np.array([float(r['R_strict']) for r in rows]),
        'val': np.array([float(r['val_acc_diff']) for r in rows]),
        'test': np.array([float(r['test_acc_diff']) for r in rows]),
    }
    np.testing.assert_allclose(d_frozen, stored['D_frozen'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r_joint, stored['R_joint'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(d_single, stored['D_single'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r_strict, stored['R_strict'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(val_diff, stored['val'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(test_diff, stored['test'], rtol=0, atol=1e-12)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(rows), size=(N_BOOT, len(rows)))
    b_df = d_frozen[idx].mean(axis=1)
    b_rj = np.exp(np.log(r_joint)[idx].mean(axis=1))
    b_ds = d_single[idx].mean(axis=1)
    b_rs = np.exp(np.log(r_strict)[idx].mean(axis=1))
    b_v = val_diff[idx].mean(axis=1)
    b_t = test_diff[idx].mean(axis=1)

    out = {
        'P1': {'mean': float(d_frozen.mean()), 'bootstrap95': pct95(b_df), 'pass': bool(np.quantile(b_df, .975) < 0)},
        'P2': {'geometric_mean_ratio': float(np.exp(np.log(r_joint).mean())), 'bootstrap95': pct95(b_rj), 'pass': bool(np.quantile(b_rj, .975) < 1.40)},
        'P3': {'mean': float(d_single.mean()), 'bootstrap95': pct95(b_ds), 'pass': bool(np.quantile(b_ds, .975) < 0)},
        'P4': {'geometric_mean_ratio': float(np.exp(np.log(r_strict).mean())), 'bootstrap95': pct95(b_rs), 'pass': bool(np.quantile(b_rs, .975) < 2.25)},
        'validation_guardrail': {'mean': float(val_diff.mean()), 'bootstrap95': pct95(b_v), 'pass': bool(np.quantile(b_v, .025) > -0.02)},
        'test_safeguard': {'mean': float(test_diff.mean()), 'bootstrap95': pct95(b_t), 'pass': bool(np.quantile(b_t, .025) > -0.02)},
    }
    out['overall'] = 'CONFIRMATORY_PASS' if all(v['pass'] for v in out.values()) else 'CONFIRMATORY_FAIL'
    print(json.dumps(out, indent=2))
    if out['overall'] != 'CONFIRMATORY_PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
