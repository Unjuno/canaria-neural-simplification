from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rows = []
with (HERE / "seed_rows.csv").open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["eligible"] != "True":
            continue
        assert row["test_evaluated_after_validation_only"] == "True"
        rows.append(row)

assert len(rows) >= 8
D = np.array([float(r["D_worst"]) for r in rows])
R = np.array([float(r["R_worst"]) for r in rows])
S = np.array([float(r["basis_sensitivity"]) for r in rows])
V = np.array([float(r["validation_accuracy_difference_worst"]) for r in rows])
T = np.array([float(r["test_accuracy_difference_worst"]) for r in rows])

rng = np.random.default_rng(20261110)
idx = rng.integers(0, len(rows), size=(100000, len(rows)))

def ci(x):
    q = np.quantile(x, [0.025, 0.975])
    return [float(q[0]), float(q[1])]

calc = {
    "P1": {"estimate": float(D.mean()), "ci95": ci(D[idx].mean(axis=1))},
    "P2": {"estimate": float(np.exp(np.log(R).mean())), "ci95": [float(x) for x in np.exp(np.quantile(np.log(R)[idx].mean(axis=1), [0.025, 0.975]))]},
    "P3": {"estimate": float(np.exp(np.log(S).mean())), "ci95": [float(x) for x in np.exp(np.quantile(np.log(S)[idx].mean(axis=1), [0.025, 0.975]))]},
    "validation_safeguard": {"estimate": float(V.mean()), "ci95": ci(V[idx].mean(axis=1))},
    "test_safeguard": {"estimate": float(T.mean()), "ci95": ci(T[idx].mean(axis=1))},
}

stored = json.loads((HERE / "RESULT.json").read_text(encoding="utf-8"))
keys = {
    "P1": "estimate_mean_D_worst",
    "P2": "estimate_geomean_R_worst",
    "P3": "estimate_geomean_S",
    "validation_safeguard": "estimate_mean_difference",
    "test_safeguard": "estimate_mean_difference",
}
for name, rec in calc.items():
    s = stored["endpoints"][name]
    assert math.isclose(rec["estimate"], s[keys[name]], rel_tol=0, abs_tol=1e-12)
    assert np.allclose(rec["ci95"], s["ci95"], rtol=0, atol=1e-12)

assert calc["P1"]["ci95"][1] < 0
assert calc["P2"]["ci95"][1] < 1.25
assert calc["P3"]["ci95"][1] < 1.15
assert calc["validation_safeguard"]["ci95"][0] > -0.03
assert calc["test_safeguard"]["ci95"][0] > -0.03
assert stored["status"] == "CONFIRMATORY_PASS"
print("C20_AUDIT_PASS")
