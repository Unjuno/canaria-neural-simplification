from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts/reproduce/core_discovery_digits/run_confirmatory.py"
SUMMARY = ROOT / "results/core_discovery_digits/confirm_summary.json"
PROTOCOL = ROOT / "results/core_discovery_digits/PROTOCOL_LOCK.json"


def pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def bootstrap_ci(values: np.ndarray, reps: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def close(a: float, b: float, tol: float = 1e-7) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    expected = json.loads(SUMMARY.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows_expected = {int(row["seed"]): row for row in expected["per_seed_selected_budgets"]}
    seeds = [int(x) for x in protocol["confirmatory_seeds"]]

    errors: list[str] = []
    rows: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="canaria-core-repro-") as tmp:
        tmpdir = Path(tmp)
        for seed in seeds:
            out = tmpdir / f"seed_{seed}.json"
            proc = subprocess.run(
                [sys.executable, str(RUNNER), "--seed", str(seed), "--out", str(out)],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            if proc.returncode != 0:
                errors.append(f"seed {seed}: runner failed: {proc.stderr[-2000:]}")
                continue
            result = json.loads(out.read_text(encoding="utf-8"))
            exp = rows_expected[seed]
            sep = result.get("selected_sep")
            comp = result.get("selected_comp")
            if not sep or not comp:
                errors.append(f"seed {seed}: missing selected endpoint")
                continue

            row = {
                "seed": seed,
                "componentwise": int(sep["budget"]),
                "composed": int(comp["budget"]),
                "log2_ratio": float(result["log2_budget_ratio"]),
                "test_acc_diff": float(result["test_acc_diff_comp_minus_sep"]),
            }
            rows.append(row)

            if row["componentwise"] != int(exp["componentwise"]):
                errors.append(
                    f"seed {seed}: componentwise budget {row['componentwise']} != {exp['componentwise']}"
                )
            if row["composed"] != int(exp["composed"]):
                errors.append(
                    f"seed {seed}: composed budget {row['composed']} != {exp['composed']}"
                )
            if not close(row["log2_ratio"], exp["log2_ratio"], 1e-12):
                errors.append(
                    f"seed {seed}: log2 ratio {row['log2_ratio']} != {exp['log2_ratio']}"
                )
            if not close(row["test_acc_diff"], exp["test_acc_diff"], 1e-6):
                errors.append(
                    f"seed {seed}: test accuracy diff {row['test_acc_diff']} != {exp['test_acc_diff']} within 1e-6"
                )

    aggregate = {}
    if len(rows) == len(seeds):
        log2s = np.array([r["log2_ratio"] for r in rows], dtype=float)
        sep_b = np.array([r["componentwise"] for r in rows], dtype=float)
        comp_b = np.array([r["composed"] for r in rows], dtype=float)
        test_d = np.array([r["test_acc_diff"] for r in rows], dtype=float)

        reps = int(protocol["primary"]["bootstrap_reps"])
        boot_seed = int(protocol["primary"]["bootstrap_seed"])
        ci = bootstrap_ci(log2s, reps, boot_seed)
        test_ci = bootstrap_ci(test_d, reps, boot_seed)
        aggregate = {
            "mean_componentwise_budget": float(sep_b.mean()),
            "mean_composed_budget": float(comp_b.mean()),
            "mean_log2_budget_ratio": float(log2s.mean()),
            "geometric_mean_budget_ratio": float(2.0 ** log2s.mean()),
            "bootstrap95_mean_log2_budget_ratio": ci,
            "mean_test_acc_diff": float(test_d.mean()),
            "bootstrap95_mean_test_acc_diff": test_ci,
            "composed_lower_count": int(np.sum(comp_b < sep_b)),
        }

        p = expected["primary"]
        s = expected["secondary_confirmatory"]
        checks = [
            ("mean componentwise budget", aggregate["mean_componentwise_budget"], p["mean_componentwise_budget"], 0.0),
            ("mean composed budget", aggregate["mean_composed_budget"], p["mean_composed_budget"], 0.0),
            ("mean log2 ratio", aggregate["mean_log2_budget_ratio"], p["mean"], 1e-12),
            ("geometric ratio", aggregate["geometric_mean_budget_ratio"], p["geometric_mean_budget_ratio"], 1e-12),
            ("bootstrap lower", ci[0], p["bootstrap95"][0], 1e-12),
            ("bootstrap upper", ci[1], p["bootstrap95"][1], 1e-12),
            ("mean test diff", aggregate["mean_test_acc_diff"], s["mean"], 1e-6),
        ]
        for label, got, want, tol in checks:
            if not close(got, want, tol):
                errors.append(f"aggregate {label}: {got} != {want} within {tol}")
        if aggregate["composed_lower_count"] != int(p["wins_composed_lower_budget"]):
            errors.append(
                f"aggregate composed-lower count: {aggregate['composed_lower_count']} != {p['wins_composed_lower_budget']}"
            )

    report = {
        "status": "PASS" if not errors else "FAIL",
        "evidence_class": "reproduction_of_existing_confirmatory_cohort",
        "source_summary": str(SUMMARY.relative_to(ROOT)),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "runner": str(RUNNER.relative_to(ROOT)),
        "git_head": git_head(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": pkg_version("numpy"),
            "torch": pkg_version("torch"),
            "scikit_learn": pkg_version("scikit-learn"),
        },
        "seeds": seeds,
        "rows": rows,
        "aggregate": aggregate,
        "errors": errors,
        "interpretation": "A PASS validates reproduction of the already-observed residual-MLP confirmatory cohort under the recorded modern pinned environment. It is not a new independent scientific replication.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": errors, "aggregate": aggregate}))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
