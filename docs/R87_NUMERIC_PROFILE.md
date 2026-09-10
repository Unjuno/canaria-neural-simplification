# R87: a tested numerical profile, not historical-value recovery

## Problem and intervention

The original scientific source remains byte-identical (Git blob `8759933ed2bb95014c3afc835acafa1743e40ea6`). Known seeds1200-1207 exposed numerical sensitivity; this work does not overwrite their historical summary or widen comparison tolerances.

R87D5 fixed ATen AVX2, MKL COMPATIBLE, one intra/inter-op thread, disabled oneDNN and enabled deterministic algorithms. It still failed Intel/AMD cross-host equality despite exact data, initial parameters, random schedules, first logits and first gradients. The first AdamW parameter update differed.

R87D6 replayed one common saved initial-parameter/gradient fixture. On identical Torch CPU library bytes and identical Python optimizer source/scalars, first/second moments matched, but float32 square-root outputs differed by at most one float32 representable step on this fixture. Both manual replay and actual AdamW updates agreed within each host. This is operator localization, not proof of the unique cause of historical archive mismatches. The initial replay metadata lookup failed after arrays had been saved; the failed source/artifact remains history, and corrected replay reports are separate.

R87D7 adds exactly one scoped intervention to D5: only float32 Tensor.sqrt calls inside AdamW.step are computed at float64 then cast back. Parameter/moment storage remains float32. Teacher loss and replacement losses remain original. The original data, fit schedule, budget grid and selection thresholds remain unchanged. This is a NEW numerical recipe.

## Implementation assumptions and variable table

The implementation uses CPU eager PyTorch2.10.0, a single-worker-process numerical context, non-fused original AdamW execution and the original model. It is not a GPU, torch.compile, multi-thread-safe global optimizer patch, or universal backend guarantee. No quantizer, zero-point or packed-weight ABI is involved.

| Symbol | Meaning (Japanese) | SI unit | Definition | Domain / assumptions | Type |
|---|---|---|---|---|---|
| t | 平方根の入力。AdamWの二次モーメント | 1 | input tensor during optimizer step | finite nonnegative float32 CPU entries in this experiment | vector/tensor |
| u | 精度を上げた中間表現 | 1 | float64 conversion of t | same shape; float32 finite values embed exactly in float64 | vector/tensor |
| r | 更新に戻す平方根結果 | 1 | float64 sqrt of u, cast to original float32 | same shape as t; finite tested inputs | vector/tensor |

Algorithm: convert t to float64, evaluate its square root, cast the result to float32, and use it at the same place in the unchanged AdamW step. All other operations are original. The wrapper restores Tensor.sqrt in a finally block and refuses a conflicting override. Model activations and losses are dimensionless numerical coordinates, not physical units; the tensor shape and returned dtype are unchanged. No universal error bound or correct-rounding theorem is claimed from the finite empirical test.

## Validated scope

D7's full8 known seeds plus2 prespecified process repeats on each of three recorded hosts passed exact outcome, data/init/RNG, teacher/fit-state and saved-array comparisons. The two GitHub hosts were AMD EPYC9V74 and the local host Intel Xeon E5-2673v4. This is30 paired/repeat executions, only8 distinct known models. The known-cohort average budgets happened to be3584/1728, but the paired geometric ratio was0.4926924861, not the old archive0.4823393150. Historical parity remains unresolved.

See `../results/reproduction/r87_sqrt_profile/VALIDATION_GATE.json`. Any fresh result is separately reported under R87R2, not inferred from known-seed agreement. R87R1 was never run because its D5 prerequisite failed; its protocol is preserved with zero fresh seeds consumed.

## Reproduce the numeric recipe

Use an isolated Python3.13.5 environment. Record your actual CPU, library build, package versions and effective MKL log; identical package labels alone are not sufficient evidence.

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5 scikit-learn==1.8.0 scipy==1.17.0 joblib==1.5.3 threadpoolctl==3.6.0
python scripts/reproduction_diagnostics/r87_sqrt_profile.py --mode suite --out /tmp/r87d7
```

Full transitive pins used by the hosted experiment are in `.github/workflows/r87-sqrt-profile.yml`. For the separately locked fresh-cohort reproduction, use `r87r2_confirm.py --mode suite --workers 2 --out /tmp/r87r2`; repeating its seeds later is technical reproduction, not a fresh scientific cohort. Never mix host repeats into the independent seed count.

## Publication and inference boundary

Three questions are distinct: whether the tested numerical recipe repeats exactly, whether compositional simplification meets its fresh scientific gates, and whether old archived numerical values have been recovered. A PASS on one does not decide another. Historical protocols/results remain untouched and Issue87 is not automatically closed. No speed, energy, RAM, model-size or broad architecture claim follows from this repair. Fixed clocks were not used because these are numerical reproducibility tests, not performance benchmarks.

## Related primary documentation

- PyTorch reproducibility notes: https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch v2.10.0 CPU unary-kernel and VML dispatch source: https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cpu/UnaryOpsKernel.cpp and https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cpu/vml.h
- Intel oneMKL CNR settings and their conditions: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2026-0/getting-started-with-conditional-numerical.html

These explain relevant implementation constraints; the observed equality/failure comes from the saved experiments, not authority alone.

## ERROR CHECK

Original source identity, process-level isolation, intervention counts, restoration on exceptions, dimensions/units, exact repeated states, and historical-versus-new-recipe distinction are checked. These are implementation/data audits, not independent scientific peer review.
