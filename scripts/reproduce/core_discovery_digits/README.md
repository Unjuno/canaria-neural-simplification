# Baseline digits reproduction

Follow [Quickstart](../../../QUICKSTART.md). `run_confirmatory.py` is the preserved outcome-producing runner. `verify_confirmatory.py` reruns the original1200–1207 cohort and compares the locked summary. It is a reproduction utility, not a new experimental protocol.

Pinned numerical environment: Python3.11, NumPy2.4.6, scikit-learn1.9.0, PyTorch2.13.0 CPU. Install CPU torch from the official CPU index before the pinned requirements. Run from the repository root. Use `--raw-dir outputs/core-cohort-raw` to retain every output/log; never overwrite original evidence.

Tests cover missing/duplicate/incomplete cohorts, exact budget mismatch, nonfinite statistics and numerical tolerances. A PASS is limited to the recorded platform and unchanged protocol; other CPU/OS/backend versions may differ.
