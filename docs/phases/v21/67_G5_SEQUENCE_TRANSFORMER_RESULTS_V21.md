# G5 Sequence Transformer Encoder Results — v21

## Question
Does the Transformer simplification recipe selected in G3 (small ViT) transfer to a non-image Transformer encoder without new compiler search?

## Protocol history
The first seed-3299 smoke attempt exceeded the execution limit before producing any result. A compute-only amendment reduced sequence length, dataset size, and model width before any outcome was observed. The scientific condition remained 4 Transformer blocks -> 2 smaller Transformer blocks, unlabeled residual-stream fitting, and tau={0,2,8}.

Pilot seed 3299 was used only to establish viability and is excluded from confirmatory inference.

Confirmatory protocol SHA256:
`708a451d0b60298e3225c5305124edb7844ac98d1e9f79fdf3e54512064bc1e8`

## Task
Fixed synthetic non-image sequence-order classification:
- sequence length 16
- vocabulary 32
- anchor tokens 1,2,3,4 each occur exactly once
- four balanced classes encode `(pos(1)<pos(2), pos(3)<pos(4))`
- 2400 train / 800 held-out examples
- data generation fixed independently of model seed

The task is deliberately controlled. It tests modality/sequence-structure transfer, not real-language generalization.

## Model and frozen G3 transfer recipe
Teacher:
- token embedding d=24
- learned CLS and positional embeddings
- 4 pre-norm Transformer encoder blocks
- 4 attention heads
- MLP width 48
- 20,836 parameters

Compiled model:
- all 4 blocks replaced by 2 Transformer blocks
- d=24, heads=4, MLP width 24
- compiler fit on 512 unlabeled sequences by residual-stream MSE
- 8,740 parameters

Whole-model parameter reduction: **58.0534%**.

No candidate grid or family-specific search was performed on confirmatory seeds.

## Seed discipline
Confirmatory queue started at seed 3300. Eligibility was baseline held-out accuracy >=0.95. The first eight eligible seeds were:

`3300, 3301, 3302, 3303, 3306, 3307, 3308, 3311`

Baseline-ineligible seeds encountered before the eighth eligible seed:

`3304, 3305, 3309, 3310`

They were excluded using the preregistered baseline-only eligibility rule, not based on simplification outcome.

## Functional confirmatory results

| condition | mean utility | 95% seed-bootstrap CI | PASS95 seeds |
|---|---:|---:|---:|
| tau=0 | **0.99184** | **[0.97986, 1.00199]** | **8/8** |
| tau=2 | 0.99757 | [0.98429, 1.00866] | 8/8 |
| tau=8 | 0.99342 | [0.98307, 1.00017] | 8/8 |

Baseline held-out accuracy mean: **0.97703**.

### Primary decision
The preregistered zero-shot criterion required the lower 95% seed-bootstrap CI of mean tau=0 utility to be >=0.95.

**Zero-shot transfer: PASS.**

This is stronger than the preceding G3 ViT result, where zero-shot failed and bounded repair was needed. Here the frozen G3 Transformer recipe transfers from image-token classification to non-image sequence classification without task repair.

The relevant scope of “zero-shot” is limited: tensor-shape/interface changes required by replacing patch embedding with token embedding are mechanical, while the compiler depth/width, residual-stream boundary, unlabeled fitting concept, and no-search rule are inherited from G3.

## q8 whole-network state-stream follow-up
Because the primary zero-shot criterion passed, the protocol permitted q8 accounting on the same eight seeds.

Codec:
- every floating state tensor quantized to signed symmetric int8
- one FP32 scale per tensor
- tensor name and shape headers charged
- parameter/state byte stream compressed with zlib level 9
- architecture/decoder implementation treated as shared code and not charged per model
- no shell repair before q8 measurement

Results:
- mean baseline q8+zlib state stream: **21,033.75 B**
- mean compiled q8+zlib state stream: **8,735.75 B**
- mean q8+zlib reduction: **58.4672%**
- 95% CI: **[58.2625%, 58.6905%]**
- raw q8 state-stream reduction: **56.0868%**
- q8 compiled/baseline utility: **0.99683**
- 95% CI: **[0.98402, 1.00741]**
- PASS95 seeds: **8/8**
- q8 / FP32 compiled fidelity: **1.00358**
- 95% CI: **[1.00097, 1.00636]**

The q8 numbers are real encoded parameter/state streams with explicit metadata, but they are not standalone executable model files: the architecture/decoder program is shared and omitted from per-model accounting.

## Interpretation
This experiment provides positive evidence for transfer at two levels:

1. **Architecture substrate:** the simplification phenomenon previously seen in residual CNNs and then small ViTs also appears in a Transformer encoder over discrete non-image sequences.
2. **Transformer-family compiler transfer:** the G3-selected 4->2 smaller-block Transformer replacement works on this sequence task without new candidate search or task repair.

Because tau=0 already passes, complexity relocation into a repaired shell cannot explain the primary result; there is no task repair in the primary condition.

## Important limitations
- The sequence task is synthetic and structurally simple.
- This is not evidence for natural-language Transformers.
- It does not test causal masking, autoregressive error accumulation, KV-cache behavior, or long contexts.
- Several model seeds failed the baseline eligibility floor, indicating nontrivial optimization variance in this small setup.
- The q8 accounting shares architecture/decoder code and therefore differs from the exact standalone 9,926-byte v19 codec claim.
- The result does not imply all Transformer blocks or tasks are similarly compressible.

## Next discriminative experiment
The highest-information next test is **G6: a small decoder-only causal language model**, with teacher-forced next-token loss, sequence-length dependence, and autoregressive generation drift measured separately. That test can distinguish generic Transformer-family transfer from the easier encoder-classification regime.
