# Recorded environment and separate provisioning recipe

**No installation, import, training, inference or synthetic test was performed to
prepare this package.** These are future reader instructions, not permission to
execute the historical experiment or instructions to use an author's machine.

## Exact historical version / numerical contract

- Linux x86_64 with effective AVX2; Python **3.13.5**.
- torch **2.10.0+cpu**, NumPy **2.3.5**, scikit-learn **1.8.0**, SciPy **1.17.0**.
- The complete recorded distribution-version recipe is `requirements.lock.txt`
  **together with the provisioning pins below**, not the lock file alone.
  MarkupSafe 3.0.3 and mpmath 1.3.0 close two transitive dependencies omitted by
  the original study's hand-written dependency file. They were recovered from
  installed distribution metadata, not from a new resolver run.
- The recipe also pins pip 26.2.2 and setuptools 78.1.0. Setuptools is a
  declared torch dependency on Python >=3.12, including this Python 3.13.5
  environment, even though the scientific runner does not import it directly.
- torch build facts recorded historically: GCC 13.3; C++17; Intel oneAPI MKL
  2024.2, product build 20240605; OpenMP 4.5; MKL-DNN 3.7.1 (disabled at runtime);
  torch build commit `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`; CUDA disabled.
  Host names, compiler installation paths and raw build strings are not exported.
- Before importing numerical libraries the runner sets:

```text
ATEN_CPU_CAPABILITY=avx2
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
MKL_DYNAMIC=FALSE
OMP_DYNAMIC=FALSE
MKL_CBWR=COMPATIBLE
```

It sets one torch intra/inter-op thread, disables mkldnn and enables deterministic
algorithms. Its scientific entry asserts the five exact versions and AVX2.
`resource`, SIGALRM and procfs RSS monitoring make this a **Linux-only** launcher,
not a Windows/macOS launcher. Source inspection does not need this environment.

The `portable_v2_sqrt64` intervention is retained: within an isolated main Python
thread, only AdamW's step temporarily replaces float32 `Tensor.sqrt` with a
float64 square root cast back to float32. Both replacements are restored even on
exceptions. It also encloses the teacher's inherited AdamW recipe. Do not use it
concurrently with other Python threads or silently substitute default float32
square root. The scientific tests include restoration and disjoint-local-update
checks; they are supplied, **not newly passed**.

## Future provisioning, if independently authorized

Create a separate Python 3.13.5 environment using an independently obtained,
verified interpreter. The package does not download an interpreter or wheels.
A version-pinned installation recipe is:

```sh
python3.13 -m venv .venv-matched
.venv-matched/bin/python -m pip install --index-url https://pypi.org/simple 'pip==26.2.2' 'setuptools==78.1.0'
.venv-matched/bin/python -m pip install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt
.venv-matched/bin/python -m pip check
```

These commands are documented only and were not run. Availability of the exact
historical artifacts at these indexes was not rechecked. An offline verified
wheelhouse is also possible. Do not relax pins merely to make resolution pass.

**Not a hermetic binary lock:** original wheel files/hashes, full interpreter
build provenance and OS image are not included. Versions and numerical modes
are exact recorded requirements; this recipe alone does not prove bitwise model
identity on another host. Future provisioning and synthetic checks need their
own record. Synthetic checks need not access real Digits; the scientific loader
uses scikit-learn's bundled Digits and has no network acquisition code.
