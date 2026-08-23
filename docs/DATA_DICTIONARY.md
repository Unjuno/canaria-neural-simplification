# Data dictionary

This file defines the recurring fields used across the public result tables. Historical files may contain additional columns; the associated phase protocol remains authoritative.

| Field / term | Meaning |
|---|---|
| `seed` | Training/model seed. For clustered inference, the trained network is the primary statistical cluster. |
| `tau` | Additional repair/adaptation budget in epochs. `tau=0` means no repair. |
| `span_start`, `span_end` | Inclusive block indices of the replaced/composed span. |
| `width` | Number of blocks in the span. |
| `utility` / `utility_ratio` | Compiled/intervention performance divided by the matched reference performance. For adaptive experiments the denominator is normally a matched continued-training control. |
| `PASS95` / `pass_rate` | Whether utility is at least 0.95; `pass_rate` is the fraction of seeds/events meeting that threshold. |
| `retention` | Performance relative to a same-seed FP32 replacement/reference, used to isolate quantization/sparsification damage from reference-fit quality. |
| `composition_gain` / `G` | Operational simplification gain comparing a parent span with its decomposed/selected representations. Positive values indicate subadditivity under the current grammar. |
| `strong_simplification` | In Phase A, a preregistered binary endpoint based on composition gain and utility requirements. |
| `Canary` | Boundary/sensor quantity measured only after the blinded Stage-1 lock in confirmatory Phase A. Current Canary is not established as a causal driver. |
| `bits` | Research quantization grid width unless a hardware datatype is explicitly stated. E.g. 4-bit does not automatically mean hardware FP4. |
| `K` | Number of retained/stored scalar coefficients in count-limited experiments. |
| `bytes` | Storage under the exact accounting rule of that phase. Check whether it is core-only, nominal, or real serialized whole-network bytes. |
| `core bytes` | Bytes for the replacement/compiled core only. Shell/head parameters are excluded. |
| `whole-network bytes` | Bytes for every model component included by the specified codec. |
| `nominal bytes` | Bit-accounting estimate from values, masks/indices, scales, and metadata; not automatically a real file size. |
| `entropy` / `ideal code length` | Information-theoretic/proxy code length; not automatically a realized serialization. |
| `combined fidelity` | Accuracy of the final compressed/quantized model divided by the repaired compiled FP32 state. |
| `compression fidelity` | Accuracy relative to a denser compressed baseline, used to isolate a specific compression stage. |
| `matched-control utility` | Accuracy of the compiled/compressed model divided by a control that received the same additional training budget. |
| `CI` | Unless a file says otherwise, later decisive experiments use seed-level bootstrap intervals rather than treating multiple spans from one network as independent. |

## Evidence labels

- **Confirmatory**: condition and decision rule fixed before outcome inspection.
- **Independent holdout**: condition selected earlier, then evaluated on fresh seeds without reselection.
- **Exploratory/pilot**: hypothesis-generating; may motivate a later holdout.
- **Negative result**: a failed hypothesis retained in the evidence chain.

## Storage interpretation rule

Never compare byte counts across phases without reading the protocol. A 44.5-byte compiled core and a 9,926-byte whole-network codec answer different questions.