import argparse, json, math
from pathlib import Path

import numpy as np

D = 32
R = 4
ALPHA = np.array([0.15, 0.12, 0.10, 0.08], dtype=np.float64)
SEEDS = list(range(103000, 103064))
REL_FROB_TOL = 1e-10
ABS_ZERO_TOL = 1e-12


def orth(rng, d=D, r=R):
    q, _ = np.linalg.qr(rng.normal(size=(d, r)))
    return q[:, :r]


def core(rng):
    q1, _ = np.linalg.qr(rng.normal(size=(R, R)))
    q2, _ = np.linalg.qr(rng.normal(size=(R, R)))
    return q1 @ np.diag(ALPHA) @ q2.T


def oracle_rank(residual):
    s = np.linalg.svd(residual, compute_uv=False)
    norm = float(np.linalg.norm(s))
    if norm <= ABS_ZERO_TOL:
        return 0, s
    for k in range(len(s) + 1):
        tail = float(np.linalg.norm(s[k:]))
        if tail / norm <= REL_FROB_TOL:
            return k, s
    return len(s), s


def make_independent(rng):
    u1, v1, u2, v2 = orth(rng), orth(rng), orth(rng), orth(rng)
    r1 = u1 @ core(rng) @ v1.T
    r2 = u2 @ core(rng) @ v2.T
    return np.eye(D) + r1, np.eye(D) + r2


def make_shared(rng):
    u, v = orth(rng), orth(rng)
    r1 = u @ core(rng) @ v.T
    r2 = u @ core(rng) @ v.T
    return np.eye(D) + r1, np.eye(D) + r2


def make_inverse(rng):
    a1, _ = make_independent(rng)
    a2 = np.linalg.inv(a1)
    return a1, a2


def one_regime(seed, name):
    offset = {"independent": 0, "shared_subspace": 1000000, "inverse_cancellation": 2000000}[name]
    rng = np.random.default_rng(seed + offset)
    if name == "independent":
        a1, a2 = make_independent(rng)
    elif name == "shared_subspace":
        a1, a2 = make_shared(rng)
    elif name == "inverse_cancellation":
        a1, a2 = make_inverse(rng)
    else:
        raise ValueError(name)

    I = np.eye(D)
    r1m = a1 - I
    r2m = a2 - I
    rc = a2 @ a1 - I
    r1, s1 = oracle_rank(r1m)
    r2, s2 = oracle_rank(r2m)
    rcomp, sc = oracle_rank(rc)
    denom = r1 + r2
    ratio = 0.0 if denom == 0 else rcomp / denom
    return {
        "seed": seed,
        "regime": name,
        "component_rank_1": r1,
        "component_rank_2": r2,
        "composed_rank": rcomp,
        "componentwise_oracle_parameters": 2 * D * denom,
        "composed_oracle_parameters": 2 * D * rcomp,
        "composed_over_componentwise_ratio": ratio,
        "condition_number_a1": float(np.linalg.cond(a1)),
        "condition_number_a2": float(np.linalg.cond(a2)),
        "composition_residual_frobenius": float(np.linalg.norm(rc, ord="fro")),
        "singular_values_component_1": s1.tolist(),
        "singular_values_component_2": s2.tolist(),
        "singular_values_composed": sc.tolist()
    }


def evaluate(rows):
    by = {k: [] for k in ["independent", "shared_subspace", "inverse_cancellation"]}
    for row in rows:
        by[row["regime"]].append(row)

    def summary(name, threshold, direction):
        vals = np.array([r["composed_over_componentwise_ratio"] for r in by[name]], dtype=np.float64)
        if direction == "ge":
            count = int(np.sum(vals >= threshold))
            passed = float(np.median(vals)) >= threshold and count >= 60
        else:
            count = int(np.sum(vals <= threshold))
            passed = float(np.median(vals)) <= threshold and count >= 60
        return {
            "n": len(vals),
            "median_ratio": float(np.median(vals)),
            "min_ratio": float(np.min(vals)),
            "max_ratio": float(np.max(vals)),
            "threshold": threshold,
            "direction": direction,
            "count_meeting_threshold": count,
            "required_count": 60,
            "pass": bool(passed),
            "rank_triplets": sorted(set((r["component_rank_1"], r["component_rank_2"], r["composed_rank"]) for r in by[name]))
        }

    s_ind = summary("independent", 0.95, "ge")
    s_shared = summary("shared_subspace", 0.55, "le")
    s_inv = summary("inverse_cancellation", 0.05, "le")
    cond_ok = all(max(r["condition_number_a1"], r["condition_number_a2"]) < 10.0 for r in rows)
    complete = all(len(by[k]) == len(SEEDS) for k in by)
    all_pass = complete and cond_ok and s_ind["pass"] and s_shared["pass"] and s_inv["pass"]
    decision = "SYNTHETIC_RANK_MECHANISM_PASS" if all_pass else "SYNTHETIC_RANK_MECHANISM_FAIL_OR_UNCERTAIN"
    return {
        "schema_version": 1,
        "experiment": "SYNTHETIC_KNOWN_RANK_NULL_V1",
        "decision": decision,
        "dimension": D,
        "component_target_rank": R,
        "relative_frobenius_tolerance": REL_FROB_TOL,
        "absolute_zero_tolerance": ABS_ZERO_TOL,
        "seed_count_per_regime": len(SEEDS),
        "complete": complete,
        "condition_number_lt_10_all": cond_ok,
        "regimes": {
            "independent": s_ind,
            "shared_subspace": s_shared,
            "inverse_cancellation": s_inv
        },
        "interpretation": "The oracle budget depends on the rank algebra of the composed residual. Independent generic components provide a null where no subadditive rank gain is expected; shared subspaces and inverse cancellation provide positive controls with known lower composed rank.",
        "boundary": "Synthetic linear residual rank oracle only; not neural-network MDL, Kolmogorov complexity, task utility, or learned-optimizer evidence."
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        for name in ["independent", "shared_subspace", "inverse_cancellation"]:
            rows.append(one_regime(seed, name))
    decision = evaluate(rows)
    (out / "ROWS.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out / "DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))
    return 0 if decision["decision"] == "SYNTHETIC_RANK_MECHANISM_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
