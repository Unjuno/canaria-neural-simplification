import csv, json
from pathlib import Path
import numpy as np

base = Path(__file__).resolve().parent
rows = list(csv.DictReader(open(base / "seed_rows.csv")))
a = np.array([
    [
        float(r["anchor_self_nmse"]) - float(r["frozen_nmse"]),
        float(r["anchor_self_nmse"]) - float(r["sketch_only_16_nmse"]),
        float(r["anchor_self_nmse"]) - float(r["generic_best_nmse"]),
        float(r["anchor_self_nmse"]) / float(r["full_64_nmse"]),
        float(r["anchor_self_val_acc"]) - float(r["full_64_val_acc"]),
    ]
    for r in rows
])
rng = np.random.default_rng(20261230)
idx = rng.integers(0, len(a), size=(100000, len(a)))
b = a[idx]
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
calc = {
    "P1": [float(a[:,0].mean()), ci(b[:,:,0].mean(1))],
    "P2": [float(a[:,1].mean()), ci(b[:,:,1].mean(1))],
    "P3": [float(a[:,2].mean()), ci(b[:,:,2].mean(1))],
    "P4": [float(np.exp(np.log(a[:,3]).mean())), ci(np.exp(np.log(b[:,:,3]).mean(1)))],
    "validation_safeguard": [float(a[:,4].mean()), ci(b[:,:,4].mean(1))],
}
stored = json.load(open(base / "RESULT.json"))["endpoints"]
assert np.allclose(calc["P1"][0], stored["P1"]["estimate_mean_D_frozen"]) and np.allclose(calc["P1"][1], stored["P1"]["ci95"])
assert np.allclose(calc["P2"][0], stored["P2"]["estimate_mean_D_sketch"]) and np.allclose(calc["P2"][1], stored["P2"]["ci95"])
assert np.allclose(calc["P3"][0], stored["P3"]["estimate_mean_D_best_generic"]) and np.allclose(calc["P3"][1], stored["P3"]["ci95"])
assert np.allclose(calc["P4"][0], stored["P4"]["estimate_geomean_R_full"]) and np.allclose(calc["P4"][1], stored["P4"]["ci95"])
assert np.allclose(calc["validation_safeguard"][0], stored["validation_safeguard"]["estimate_mean_difference"]) and np.allclose(calc["validation_safeguard"][1], stored["validation_safeguard"]["ci95"])
print("C23_AUDIT_PASS")
