from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


torch.set_num_threads(1)

SEEDS = (2300, 2301, 2302)
RECIPES = (
    {"id": "baseline_60", "epochs": 60, "lr": 2e-3, "weight_decay": 1e-4, "batch_size": 64},
    {"id": "short_15", "epochs": 15, "lr": 2e-3, "weight_decay": 1e-4, "batch_size": 64},
    {"id": "short_25", "epochs": 25, "lr": 2e-3, "weight_decay": 1e-4, "batch_size": 64},
    {"id": "short_40", "epochs": 40, "lr": 2e-3, "weight_decay": 1e-4, "batch_size": 64},
    {"id": "reg_40", "epochs": 40, "lr": 1e-3, "weight_decay": 1e-3, "batch_size": 64},
    {"id": "reg_60", "epochs": 60, "lr": 1e-3, "weight_decay": 1e-3, "batch_size": 64},
    {"id": "reg_100", "epochs": 100, "lr": 1e-3, "weight_decay": 1e-3, "batch_size": 64},
    {"id": "strongreg_40", "epochs": 40, "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 64},
    {"id": "strongreg_80", "epochs": 80, "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 64},
    {"id": "low_lr_80", "epochs": 80, "lr": 5e-4, "weight_decay": 1e-3, "batch_size": 64},
    {"id": "low_lr_120", "epochs": 120, "lr": 5e-4, "weight_decay": 1e-3, "batch_size": 64},
    {"id": "fullbatch_200", "epochs": 200, "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 512},
    {"id": "fullbatch_400", "epochs": 400, "lr": 5e-4, "weight_decay": 1e-2, "batch_size": 512},
)


class ResBlock(nn.Module):
    def __init__(self, d: int = 64):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 128)
        self.fc2 = nn.Linear(128, d)

    def forward(self, x):
        z = self.ln(x)
        z = F.gelu(self.fc1(z))
        z = self.fc2(z)
        return x + z


class Net(nn.Module):
    def __init__(self, seed: int, input_dim: int, d: int = 64, depth: int = 4):
        super().__init__()
        torch.manual_seed(seed)
        self.stem = nn.Linear(input_dim, d)
        self.blocks = nn.ModuleList([ResBlock(d) for _ in range(depth)])
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        h = F.gelu(self.stem(x))
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


def train_val_data():
    X, y = load_diabetes(return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    idx = np.arange(len(y))

    # Preserve the completed Phase 3 split exactly, but Stage A never materializes
    # or evaluates the test targets after they are split off.
    train_idx, _test_idx = train_test_split(idx, test_size=0.25, random_state=2234)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=6678)

    x_mean = X[train_idx].mean(axis=0, keepdims=True)
    x_std = X[train_idx].std(axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    y_mean = float(y[train_idx].mean())
    y_std = float(y[train_idx].std())
    if y_std < 1e-8:
        y_std = 1.0

    Xn_train = (X[train_idx] - x_mean) / x_std
    Xn_val = (X[val_idx] - x_mean) / x_std
    yn_train = (y[train_idx] - y_mean) / y_std
    yn_val = (y[val_idx] - y_mean) / y_std

    return (
        torch.tensor(Xn_train, dtype=torch.float32),
        torch.tensor(yn_train, dtype=torch.float32),
        torch.tensor(Xn_val, dtype=torch.float32),
        torch.tensor(yn_val, dtype=torch.float32),
        int(X.shape[1]),
        {"n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test_held_out": int(len(_test_idx))},
    )


def train_model(seed: int, Xt, yt, input_dim: int, recipe: dict):
    model = Net(seed, input_dim=input_dim)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"])
    )
    gen = torch.Generator().manual_seed(seed + 999)
    batch = int(recipe["batch_size"])
    for _ in range(int(recipe["epochs"])):
        perm = torch.randperm(len(Xt), generator=gen)
        for i in range(0, len(Xt), batch):
            ix = perm[i : i + batch]
            opt.zero_grad()
            loss = F.mse_loss(model(Xt[ix]), yt[ix])
            loss.backward()
            opt.step()
    return model


def mse(model, X, y):
    with torch.no_grad():
        return float(F.mse_loss(model(X), y))


def r2(model, X, y):
    with torch.no_grad():
        pred = model(X)
        sse = float(((pred - y) ** 2).sum())
        sst = float(((y - y.mean()) ** 2).sum()) + 1e-12
        return 1.0 - sse / sst


def run():
    Xt, yt, Xv, yv, input_dim, data_meta = train_val_data()
    rows = []

    for recipe_index, recipe in enumerate(RECIPES):
        for seed in SEEDS:
            model = train_model(seed, Xt, yt, input_dim, recipe)
            rows.append(
                {
                    "recipe_index": recipe_index,
                    "recipe_id": recipe["id"],
                    "seed": seed,
                    "train_mse": mse(model, Xt, yt),
                    "validation_mse": mse(model, Xv, yv),
                    "validation_r2": r2(model, Xv, yv),
                }
            )

    aggregates = []
    for recipe_index, recipe in enumerate(RECIPES):
        rr = [row for row in rows if row["recipe_id"] == recipe["id"]]
        vals = np.asarray([row["validation_r2"] for row in rr], dtype=np.float64)
        aggregates.append(
            {
                "recipe_index": recipe_index,
                "recipe": dict(recipe),
                "mean_validation_r2": float(vals.mean()),
                "std_validation_r2": float(vals.std(ddof=0)),
                "min_validation_r2": float(vals.min()),
                "max_validation_r2": float(vals.max()),
            }
        )

    baseline = next(x for x in aggregates if x["recipe"]["id"] == "baseline_60")
    baseline_mean = baseline["mean_validation_r2"]
    for x in aggregates:
        x["improvement_over_baseline_mean_r2"] = float(x["mean_validation_r2"] - baseline_mean)
        x["eligible"] = bool(
            x["mean_validation_r2"] >= 0.35
            and x["improvement_over_baseline_mean_r2"] >= 0.10
        )

    eligible = [x for x in aggregates if x["eligible"]]
    eligible.sort(
        key=lambda x: (
            -x["mean_validation_r2"],
            x["std_validation_r2"],
            x["recipe_index"],
        )
    )
    chosen = eligible[0] if eligible else None

    return {
        "stage": "A_teacher_only_exploration",
        "evidence_class": "calibration_only_not_confirmatory_evidence",
        "test_evaluated": False,
        "replacement_fitting_performed": False,
        "seeds": list(SEEDS),
        "data": data_meta,
        "architecture_unchanged_from_completed_phase3": True,
        "selection_gate": {
            "mean_validation_r2_gte": 0.35,
            "improvement_over_baseline_mean_validation_r2_gte": 0.10,
            "tie_break": "higher mean validation R2, then lower population std, then earlier locked recipe order",
        },
        "per_seed_rows": rows,
        "recipe_aggregates": aggregates,
        "baseline_mean_validation_r2": baseline_mean,
        "stage_a_status": "PASS_SELECT_RECIPE" if chosen is not None else "STOP_NO_ELIGIBLE_RECIPE",
        "chosen_recipe": None if chosen is None else chosen["recipe"],
        "chosen_recipe_mean_validation_r2": None if chosen is None else chosen["mean_validation_r2"],
        "chosen_recipe_improvement_over_baseline": None if chosen is None else chosen["improvement_over_baseline_mean_r2"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "stage_a_status": result["stage_a_status"],
        "baseline_mean_validation_r2": result["baseline_mean_validation_r2"],
        "chosen_recipe": result["chosen_recipe"],
        "chosen_recipe_mean_validation_r2": result["chosen_recipe_mean_validation_r2"],
        "chosen_recipe_improvement_over_baseline": result["chosen_recipe_improvement_over_baseline"],
        "test_evaluated": result["test_evaluated"],
    }))
