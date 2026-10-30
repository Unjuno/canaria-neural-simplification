# Environment: exact inherited matched recipe

No new packages are introduced beyond the pinned sibling matched package.
`DEPENDENCY.json` binds its entire 17-file source/doc closure, including its
[environment recipe](../digits_matched_source_20261027/ENVIRONMENT.md) and
[requirements version file](../digits_matched_source_20261027/requirements.lock.txt).

Scientific runtime is Linux x86_64, effective AVX2, Python 3.13.5,
torch 2.10.0+cpu, NumPy 2.3.5, scikit-learn 1.8.0 and SciPy 1.17.0.
The full dependency recipe also records joblib, threadpoolctl, sympy, mpmath,
networkx, filelock, fsspec, Jinja2, MarkupSafe and typing_extensions. Provisioning
pins pip 26.2.2 and setuptools 78.1.0 in the matched ENVIRONMENT.md; these are
part of the recipe, not optional omissions from requirements.lock.txt.

The matched helper sets the exact single-thread AVX2/MKL environment before
numerical import, disables mkldnn, enables deterministic algorithms, and retains
the scoped float64-square-root AdamW intervention. The grid's scientific entry
asserts the five exact runtime versions and AVX2. SIGALRM, procfs RSS and resource
usage make this launcher Linux-only.

Do not replace these requirements with a looser general environment. Versions
alone do not prove binary identity: original wheel hashes, interpreter build and
OS image are absent. Provisioning/download/install requires its own authority;
none occurred during this source export. Standard-library parent-contract tests
require no numerical dependencies. Tensor tests require a separately reviewed
existing environment and do not establish a scientific replication.
