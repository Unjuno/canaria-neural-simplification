# Canaria Phase 2 — Precision × Composition Experiments

Date: 2026-08-26

This bundle starts a new research phase. The frozen v0.2.0 public snapshot is not modified.

## Phase 2A — precision × composition frontier

**H:** At 4-bit signed uniform post-training quantization, the composed replacement retains a lower minimum passing coded size than component-wise replacement.

**T:** 8 fresh seeds (31000–31007), residual-MLP digits setup, exact equal parameter count at each budget, 600 fit updates/map, validation-only selection. Precisions: 32/12/8/6/4/3 bits. Per-matrix FP16 scale for quantized conditions.

**D:** PASS iff >=7/8 coded-size wins and bootstrap95 upper bound of mean log2 coded-size ratio < 0.

**Result:** **PASS**. At 4 bit: 8/8 coded-size wins, mean log2 ratio -0.604813, bootstrap95 [-0.875351, -0.303519], geometric coded-size ratio 0.6576×. Mean selected test-accuracy difference (composed - component-wise) +0.056 percentage points; bootstrap95 [-0.222, +0.333] pt.

Endpoint coverage by precision: 32b sep 8/8, comp 7/8; 12b 8/8,7/8; 8b 8/8,7/8; 6b 7/8,7/8; 4b 8/8,8/8; 3b 0/8,0/8.

**C (counter-hypothesis / failure mode):** low precision could erase the compositional advantage or prevent either condition from satisfying fidelity. The 3-bit condition supports the latter failure mode.

**U:** n=8; one non-monotonic seed (31003) failed composed endpoint at 32/12/8 but passed at 4-bit, consistent with quantization-induced regularization or a decision-boundary perturbation. This requires a dedicated mechanism test before causal interpretation.

## Phase 2B — 3-bit capacity rescue

**H:** The 3-bit failure up to 6144 params is mainly capacity-limited; composed replacements should recover a passing endpoint by 16384 params in >=6/8 fresh seeds.

**T:** Fresh seeds 31100–31107, 3-bit per-matrix quantization, budgets 6144/8192/12288/16384 params, same validation criteria.

**D:** PASS >=6/8 composed rescues; FAIL <=2/8.

**Result:** **FAIL**. Composed rescue 0/8; component-wise rescue 0/8. At 16384 params, mean NMSE component-wise 0.1139, composed 0.0917; minimum composed NMSE 0.0820, still above the 0.08 threshold.

**C:** Increasing weight count cannot overcome a quantization error floor under coarse per-matrix scaling.

**U:** Only one quantizer family and fixed 600-update fitting were tested. Capacity and optimization effort are not fully separable.

## Phase 2C — 3-bit scale granularity

**H:** 3-bit failure is largely scale-granularity limited; row-wise FP16 scaling rescues composed replacements in >=6/8 fresh seeds at 16384 weights.

**T:** Fresh seeds 31200–31207, fixed 16384 weights, compare per-matrix vs per-output-channel(row-wise) symmetric 3-bit quantization.

**D:** PASS >=6/8 composed row-wise passes; FAIL <=2/8.

**Result:** **PASS**. Per-matrix pass: component-wise 0/8, composed 2/8. Row-wise pass: component-wise 7/8, composed 7/8. Mean row-wise NMSE: component-wise 0.0600, composed 0.0496. Coded size including FP16 scales: component-wise 6656 B, composed 6528 B.

**Interpretation:** 3-bit failure is not a simple precision floor or a simple capacity floor. It depends strongly on quantizer granularity. Composition still gives lower functional error on average, but row-wise scaling rescues both topologies, so the dominant mechanism here is quantizer representation rather than composition alone.

## Cross-phase conclusion

1. The compositional simplification advantage survives 4-bit post-training quantization on this residual-MLP task, but its magnitude shrinks relative to higher precision.
2. Naive 3-bit per-matrix quantization fails even after increasing replacement capacity from 6144 to 16384 parameters.
3. 3-bit row-wise scaling restores the passing criterion in 7/8 fresh seeds for both composed and component-wise replacements.
4. Therefore, bit width alone is an insufficient descriptor of deployable complexity. Scale granularity, scale-metadata cost, downstream sensitivity, and functional boundary must be accounted jointly.

## Next falsifiable question

Does quantization-aware fine-tuning recover 3-bit fidelity with much lower scale metadata than row-wise scaling, and does the composed condition require fewer repair updates or fewer scale groups?

## Repository provenance

This Phase 2 record is intentionally separate from the frozen v0.2.0 public snapshot.

- Source evidence bundle SHA256: `55fd1d94f63773f888a59074227cffc3e0814abace5717f6e6e84b340121cb6c`
- The source bundle contained transient `__pycache__` files; they are intentionally excluded from GitHub.
- Historical execution scripts used `/mnt/data/...` paths. The public runners are path-portable refactors rather than claims about the historical script bytes.
- The portable Phase 2C refactor was checked on seed 31200 and produced an exactly equal JSON object to the recorded source result.

## Public layout

- `docs/phase2/PRECISION_COMPOSITION.md` — scientific report and interpretation.
- `results/phase2/precision_composition/` — protocol locks and recorded summaries/data.
- `scripts/phase2/precision_composition/` — portable runners.
