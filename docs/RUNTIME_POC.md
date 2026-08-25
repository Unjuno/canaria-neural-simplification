# Minimal runtime materialization proof of concept

## Purpose

This is the smallest systems experiment motivated by the compact-functional-representation application idea.

It does **not** claim that Canaria is a production runtime or that runtime compilation is universally faster. It asks a narrower question:

> Can one trained compact Canaria replacement be serialized as a smaller execution artifact, materialized independently, and executed directly without reconstructing the original larger model?

## Setup

The PoC uses G7 fresh confirmatory seed 4300.

Two artifacts are created from the same training trajectory:

- **large**: the original 4-block, MLP-48 model at epoch 12;
- **compact**: the progressive `4→3→2` result, represented as the shared LM shell plus the learned 2-block compiler.

Each artifact contains:

- a PyTorch `state_dict`;
- a small JSON manifest describing the execution shape.

The compact artifact is materialized into a model that executes the learned 2-block compiler **natively**. It does not expand back into the original 4-block network.

This is therefore a minimal example of native compact execution, not a sophisticated hardware-specific compiler IR.

## Reproduce

Install the pinned reproduction environment:

```bash
python -m pip install -r scripts/reproduce/g7_confirmatory/requirements.txt
```

Run:

```bash
python scripts/reproduce/g7_confirmatory/runtime_poc.py \
  --seed 4300 \
  --out-dir runtime_poc_out \
  --repeat 5
```

The benchmark probes each saved artifact in a fresh Python process five times. CPU inference timing is averaged over 20 batch-128 forward passes after warmup in each probe.

## Recorded result

Environment:

- Python 3.13.5
- PyTorch 2.10.0+cpu
- NumPy 2.3.5
- scikit-learn 1.8.0
- psutil 7.2.2
- Linux x86_64 CPU environment

### Serialized artifact size

- large: **110,093 bytes**
- compact: **54,646 bytes**
- reduction: **50.36%**

### Parameters

- large: **23,138**
- compact: **11,042**
- reduction: **52.28%**

### CPU batch-128 inference

Five fresh-process probes:

- large mean: **47.05 ms**
- compact mean: **23.11 ms**
- compact / large: **0.491×**

Median ratio was **0.473×**.

This supports a CPU execution benefit for this specific small reduced operator. It is not evidence of universal GPU/LLM/runtime speedup.

### Load / materialization

Five fresh-process probes:

- large mean: **7.85 ms**
- compact mean: **5.86 ms**
- compact / large: **0.746×**

Auxiliary runs showed meaningful cache/filesystem sensitivity, including a run where the two load times were approximately equal. Therefore load/materialization speed is treated as a **secondary, environment-sensitive observation**, not a stable project-level claim.

### Host RSS delta

- large mean RSS increase: **4.72 MB**
- compact mean RSS increase: **4.56 MB**
- compact / large: **0.966×**

This does **not** demonstrate meaningful RAM reduction. Process/runtime allocator overhead dominates at this model scale.

### Task utility

- large test PPL: **19.2784**
- compact test PPL: **18.9322**

The compact model is not merely a lossy smaller runtime artifact in this seed; it is the already-confirmed progressive-training result. This systems PoC should not be interpreted as a new utility experiment.

## What this PoC establishes

For this small CPU setting:

1. a compact consolidated model can be serialized independently;
2. it can be materialized from a manifest/state artifact;
3. it can execute the compact operator directly rather than reconstructing the original larger network;
4. the serialized artifact is substantially smaller;
5. CPU execution is substantially faster in the measured batch-128 benchmark.

## What it does not establish

- general wall-clock speedup across hardware;
- GPU or accelerator benefit;
- meaningful peak-RAM reduction;
- cold-start benefit at large scale;
- superiority over TensorRT, ONNX, torch.compile, weight streaming, or other deployment systems;
- that this state-dict-plus-manifest format is the final Canaria IR;
- that large pretrained LLMs exhibit the same deployment behavior.

The main value of this PoC is separation: **storage/distribution and direct CPU execution improved here, while host-RAM benefit did not materially appear.**

Machine-readable results are in `results/reproduction/runtime_poc_seed4300_report.json`.
