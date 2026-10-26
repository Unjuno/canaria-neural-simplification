# California runner packaging repair (local, no CI)

The publication candidate contained the Phase 4 California wrappers but omitted
`scripts/phase3/regression_external_validity/run_seed.py`, which both wrappers
load directly. This repair restores that file **byte for byte**, not a new model
implementation:

- Source commit: `f7d23c9dad2788e64b94f2a268fe3e0b1084d77c` (`origin/phase4` at repair time).
- Git blob: `8dd3ddefb59ba250f5deca2d98d04fbff0055748` (the pin recorded in both wrappers).
- File SHA-256: `701ae354f627cf9cf7612502749e6305c67f92eb2dbc6183887a75eaf4b99fae`.

The Stage B protocol specifies the architecture and numerical profile; it does
not itself contain the base-runner blob pin. The restored base imports only
standard-library modules, NumPy, PyTorch, and scikit-learn; it requires no further
repository-relative imports. Wrappers, protocols, results, thresholds, claim
registry, and workflows are unchanged.

## Offline local smoke check

From the repository root, use an **already provisioned** Python environment
(`PYTHON` below). This command neither installs dependencies nor downloads data,
trains models, invokes a workflow, or writes experimental results. The isolated
cache is deliberately empty. It imports the wrapper and its actual base module,
without calling their main/data/training functions.

```bash
PYTHON=/path/to/already-provisioned-environment/bin/python
scratch=$(mktemp -d)
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  XDG_CACHE_HOME="$scratch/cache" SKLEARN_DATA="$scratch/sklearn" \
  TORCH_HOME="$scratch/torch" MPLCONFIGDIR="$scratch/matplotlib" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ATEN_CPU_CAPABILITY=avx2 MKL_CBWR=COMPATIBLE \
  "$PYTHON" - <<'PY'
import hashlib, importlib.util, socket
from pathlib import Path
def deny_network(*args, **kwargs):
    raise RuntimeError("Network forbidden during packaging import check")
socket.socket.connect = deny_network
socket.create_connection = deny_network
p = Path('scripts/phase3/regression_external_validity/run_seed.py')
b = p.read_bytes()
assert hashlib.sha1(b'blob ' + str(len(b)).encode() + b'\0' + b).hexdigest() == '8dd3ddefb59ba250f5deca2d98d04fbff0055748'
wrapper = Path('scripts/phase4/run_california_confirm_seed.py')
for source in (p, wrapper):
    compile(source.read_bytes(), str(source), 'exec')
s = importlib.util.spec_from_file_location('california_smoke', wrapper)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
base = m.load_base()
assert all(callable(getattr(base, name)) for name in
           ('Net', 'TinyRes', 'PairReplacedNet', 'acts', 'fit_map', 'r2', 'mse'))
print('PASS: pinned source and real wrapper/base import; no experiment run')
PY
# The temporary cache can be removed after inspection.
```

The recorded numerical profile is Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5,
scikit-learn 1.8.0, and SciPy 1.17.0. Import success under another environment is
not numerical reproduction. An actual authorized reproduction would need the
locked dataset array hashes, split, CPU profile, and existing seed policy; the
wrapper's dataset loader can download data when its cache is absent. Do not use
an empty-cache training invocation as an offline check.

## Validation of this repair

- Restored blob and SHA-256 checked against the retained Git source.
- Base and both California wrappers compile; no relative import dependencies found.
- All 18 existing lightweight unit tests pass (`PYTHONDONTWRITEBYTECODE=1
  PYTHONPATH=src python -m unittest discover -s tests -v`).
- Real imports of both wrappers and their base pass using the already provisioned
  environment with all five recorded versions above and isolated empty caches;
  socket connection attempts were blocked during the check. No training/data
  functions were invoked and no packages were installed.
- Repository integrity audit: `python tools/audit_repo.py` reports `AUDIT PASS`.
- No training, dataset download, new scientific evidence, CI, push, or merge.

This fixes a missing tracked dependency only. It does not establish full
candidate reproducibility, alter scientific claims, or clear publication gates.
