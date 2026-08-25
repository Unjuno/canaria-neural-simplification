# Canaria — Application directions and evidence status

## Core systems idea

Canaria suggests treating a trained model not only as a tensor checkpoint, but as a potentially smaller **task-conditioned functional representation** that can later be serialized, materialized, compiled, or executed directly.

Conceptually:

```text
training / consolidation
        ↓
compact functional representation
        ↓
serialize / distribute / archive
        ↓
load / materialize / compile
        ↓
execute
```

This differs from the conventional assumption:

```text
checkpoint
   ↓
load original parameterization
   ↓
execute original parameterization
```

The application space must be separated into **measured evidence** and **future engineering hypotheses**.

---

## What has now been measured

A bounded CPU-only PoC is documented in `RUNTIME_POC.md` using G7 seed 4300.

The compact artifact is a serialized learned 2-block representation (`state_dict + manifest`) and executes directly. It does **not** reconstruct the original 4-block model.

Measured results:

| metric | large | compact |
|---|---:|---:|
| serialized artifact + manifest | 110,093 B | **54,646 B** |
| parameters | 23,138 | **11,042** |
| batch-128 CPU inference, 5 fresh-process probes | 47.05 ms | **23.11 ms** |
| load/materialize, mean | 7.85 ms | 5.86 ms |
| process RSS delta | 4.72 MB | 4.56 MB |
| test PPL | 19.2784 | **18.9322** |

Supported at this PoC scope:

- **storage/distribution size reduction**;
- **native execution of the compact learned representation**;
- **lower CPU inference latency in the measured setup**.

Boundary results:

- load/materialization was lower on average but showed cache sensitivity, so it is secondary evidence;
- meaningful host-RAM reduction was **not demonstrated**;
- GPU, NPU, browser, large-model, energy, and universal runtime claims are not established.

See:

- `docs/RUNTIME_POC.md`
- `scripts/reproduce/g7_confirmatory/runtime_poc.py`
- `results/reproduction/runtime_poc_seed4300_report.json`

---

# Application directions

## 1. Compact model distribution

Instead of distributing the original full tensor checkpoint, distribute a smaller learned functional representation.

Potential benefits:

- smaller model downloads;
- lower model-registry/CDN bandwidth;
- cheaper cross-region replication;
- smaller application/container artifacts;
- lower archival storage.

The small CPU PoC provides initial evidence for this direction through a 50.36% reduction in serialized artifact+manifest size. Generalization to larger models remains open.

---

## 2. Native compact execution

The strongest deployment form is:

```text
compact representation
→ materialize compact operator
→ execute compact operator directly
```

rather than:

```text
compact representation
→ reconstruct original large model
→ execute original model
```

Potential representations include:

- reduced-depth spans;
- fused learned operators;
- low-rank or factored operators;
- structured sparse kernels;
- small replacement networks;
- analytic/FIR-like replacements where valid;
- hardware-specific generated kernels.

The current PoC demonstrates direct execution for one reduced-depth learned replacement on CPU.

---

## 3. Spanwise / just-in-time materialization

A future runtime could keep a model compact and materialize only the span currently needed:

```text
compact model
   ↓
materialize span 1 → execute → release
   ↓
materialize span 2 → execute → release
   ↓
...
```

Potential peak memory could approach:

```text
compact representation
+ current materialized span
+ activations / KV cache
```

rather than the entire expanded weight set.

This remains **unproven**. The current PoC did not demonstrate meaningful host-RAM reduction.

---

## 4. Cold-start / scale-to-zero inference

A smaller functional artifact could reduce some combination of:

1. download time;
2. deserialization;
3. allocation;
4. device transfer;
5. first-token latency.

Possible settings:

- serverless inference;
- bursty autoscaling;
- short-lived workers;
- edge services.

The correct benchmark is total startup cost, not file size alone. The current load/materialization result is too environment-sensitive for a general cold-start claim.

---

## 5. Memory-bandwidth-bound inference

If a composed span is replaced by a smaller directly executable operator, the relevant gain may be **bytes moved per token** rather than only parameter count.

Future measurements should include:

- DRAM/HBM bytes per token;
- host-to-device traffic;
- tokens/s at small batch size;
- power/energy per token.

Not yet measured.

---

## 6. Edge / mobile / browser deployment

A device could carry:

- a compact Canaria representation;
- a small runtime/compiler;
- hardware-specific materialization rules.

Potential targets:

- mobile;
- embedded GPU/NPU;
- robotics;
- offline appliances;
- WebGPU/browser runtimes;
- local assistants.

Not yet validated.

---

## 7. Hardware-specific recompilation

One functional representation could potentially target multiple execution backends:

```text
same functional IR
     ├── GPU fused kernel
     ├── CPU vectorized operator
     ├── NPU graph
     └── low-memory streaming implementation
```

This would make the artifact closer to a compiler IR than a conventional checkpoint.

Not yet validated.

---

## 8. Multi-model serving

A server could retain:

- shared runtime/compiler code;
- compact model-specific representations;
- only frequently used compiled spans in a cache.

Possible benefit:

- more models per node;
- lower inactive-model storage footprint;
- faster swapping if materialization is cheap enough.

Not yet validated.

---

## 9. Compact specializations

A future deployment could potentially store:

```text
shared base
+ compact specialization A
+ compact specialization B
+ compact specialization C
```

rather than several complete checkpoints.

Possible uses:

- domain variants;
- customer-specific models;
- language variants;
- policy/safety variants;
- device-specific variants.

Current evidence does not establish clean composability of such specializations.

---

## 10. Checkpoint archival

Functional representations may be useful for:

- long-term checkpoint storage;
- preserving many intermediate training states;
- reducing experiment-archive size;
- storing model variants.

This application can tolerate expensive offline compilation if fidelity is explicit.

Only the small PoC's artifact-size result is currently measured.

---

## 11. Progressive compilation during training

Canaria also suggests a training-system application:

1. identify a consolidatable span;
2. transfer it into a smaller replacement;
3. commit the replacement;
4. continue task learning;
5. repeat.

Potential consequences:

- learning-time capacity can exceed final deployment capacity;
- later training operates on smaller structures;
- final architecture need not be fixed at initialization.

This direction is supported scientifically by the G7–G20 training-time consolidation program, though system-level training cost savings are not yet established.

---

## 12. Self-recompiling models

A longer-term loop is:

```text
learn
↓
measure functional redundancy
↓
propose compiled span
↓
estimate task risk
↓
commit
↓
recontract
↓
repeat
```

The controller cannot safely use one approximation-error threshold. Current evidence suggests it may need:

- compiler difficulty;
- residual-error direction;
- downstream task sensitivity;
- immediate task-damage estimate;
- remaining learning horizon;
- expected recovery;
- marginal compilation cost.

This remains a research direction.

---

# Deployment modes

## Mode A — Load-time materialization

```text
compact artifact → executable compact model
```

The current PoC is closest to this mode.

## Mode B — Spanwise JIT materialization

```text
compact artifact → one span → execute → release
```

Potential RAM/VRAM benefit; untested.

## Mode C — Native compact execution

```text
compact operator → execute directly
```

Demonstrated only for the small CPU PoC.

## Mode D — Hardware-adaptive recompilation

```text
functional IR → implementation selected for hardware / memory / workload
```

Untested.

---

# What future systems work should measure

Do not report parameter reduction alone. Separate:

- serialized bytes;
- compressed bytes;
- download latency;
- materialization/compile latency;
- peak CPU RAM;
- peak GPU VRAM;
- host-to-device bytes;
- memory-bandwidth traffic;
- inference latency;
- throughput;
- energy;
- task utility;
- functional fidelity;
- cache behavior.

The central engineering question is:

> Does the reduction in stored, transferred, or executed computation exceed the runtime cost of materializing or compiling it?

---

# Interpretation boundaries

Do **not** currently claim:

- that Canaria always improves inference latency;
- that compact functional representations always reduce peak RAM/VRAM;
- that runtime materialization is universally cheaper than checkpoint loading;
- that the small CPU PoC transfers directly to GPUs or large LLMs;
- that compact representations are automatically hardware-efficient;
- that functional simplification is equivalent to lossless tensor compression.

The current safe statement is narrower:

> In one small CPU PoC, a progressively consolidated learned representation was about 50% smaller on disk and about 2× faster for the measured batch-128 inference workload, while meaningful RSS reduction was not demonstrated.

---

# Suggested long-term artifact

A future deployable package might look like:

```text
model.canaria
compiler/
runtime/
manifest.json
fidelity_contract.json
hardware_profiles/
```

where `model.canaria` is a serialized functional representation rather than necessarily the original parameter tensors.
