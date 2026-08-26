# Residual-MLP direct composition reproduction

This is the smallest public runner that directly tests Canaria's central empirical pattern.

```bash
python -m pip install numpy torch scikit-learn
python scripts/reproduce/core_discovery_digits/run_confirmatory.py \
  --seed 1200 \
  --out /tmp/canaria_seed1200.json
```

Recorded confirmatory seed `1200`:

```text
component-wise selected budget = 3072
composed selected budget       = 1536
log2 ratio                     = -1.0
```

The runner uses the same validation-only minimum-passing-budget rule as the locked residual-MLP replication. The test set is not used to choose the budget.

For the full locked 8-seed result, see [`../../../results/core_discovery_digits/confirm_summary.json`](../../../results/core_discovery_digits/confirm_summary.json).

For interpretation and the fixed-2048-parameter functional-boundary control, see [`../../../docs/CORE_DISCOVERY_REPLICATION_DIGITS.md`](../../../docs/CORE_DISCOVERY_REPLICATION_DIGITS.md).