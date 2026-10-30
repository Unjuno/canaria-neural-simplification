# Environment contract, not an installation or execution receipt

Historical runtime: Linux CPU, Python **3.13.5**, torch **2.10.0+cpu**, NumPy
**2.3.5**, scikit-learn **1.8.0**, SciPy **1.17.0**. requirements.lock.txt records
version pins, not a transitive wheel/hash lock, a container image or a promise
that package indexes remain available. Provisioning/installations are separate
explicitly authorized work; none was performed for this package.

Before numerical imports the worker sets ATEN_CPU_CAPABILITY=avx2,
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1,
NUMEXPR_NUM_THREADS=1, MKL_DYNAMIC=FALSE, OMP_DYNAMIC=FALSE and
MKL_CBWR=COMPATIBLE. Then it requires AVX2, one intra/inter-op thread, disables
mkldnn, enables deterministic algorithms and uses eval/no_grad CPU eager graphs.
No CUDA/device move, compile, quantization, affinity change or tuning is present.
Optimized `python -O` is rejected because historical integrity assertions matter.

Formal worker imports and all correctness/setup fall inside the supervisor window.
The portable worker sets its 4GiB address-space limit before numerical imports;
the old module imported numerical libraries before setting that limit. This is
an explicit fail-closed operational change, not an inference/timing graph change.
An incompatible environment fails; it does not silently relax the profile.

Python standard-library synthetic controls need no NumPy/PyTorch import. Their
execution has not occurred during packaging. Numerical model/data validation,
formal timing and original-environment equivalence remain unestablished here.

Matching versions/flags cannot reproduce shared-host contention, scheduler state,
CPU frequency, affinity or other timing conditions. No historical hardware/build
identity or raw host/process record is disclosed by this source package. Future
raw environment evidence is captured privately, never presumed publishable.
