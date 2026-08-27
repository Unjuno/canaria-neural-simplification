# Systems S3 — constrained-memory feasibility for learned payload streaming

Status: **PASS** under the locked S3 memory ceiling, with a pre-outcome payload-identity amendment.

This experiment asks whether the learned G7 seed-4300 compiler payload can be processed under an explicit Linux process address-space ceiling in a regime where retaining the same logical payload fully resident cannot.

## Protocol and amendment

The original S3 protocol was committed in `79009b3da4451e10e06f52b63f505e9e05f359d7` before outcomes. It fixed:

- 4,096 logical learned-block chunks;
- S2 expected checksum `234473.0256500244`;
- Linux `RLIMIT_AS`;
- exactly **64 MiB** additional address-space headroom after import/warmup;
- full-unconstrained, full-constrained, and streaming-constrained modes;
- the PASS rule.

Before any constrained outcome was inspected, the original byte-file SHA identity check was found unsuitable because regenerated `torch.save` container bytes were not deterministic. The exact G7 reconstruction nevertheless reproduced source test PPL `18.932213342799887` and the S2 aggregate payload checksum exactly.

Commit `8c8a315cee24dddd69c701a601b0152763e0349f` therefore amended **only payload identity verification** to a deterministic tensor-content SHA256. The 64 MiB ceiling, chunk count, checksum, modes, and PASS rule were unchanged. The amended runner was committed in `1a5bf2f231244b40eab1bcf9f93ab7f03d235ba3` with blob `985b829a7a805c860e056b2dfcac5776dc5bd6e8`.

Locked tensor-content hashes:

- block 0: `78fd7f52ef6f019f6ede72c73b4928b0482d61e7f7914de532ebc084779fce56`
- block 1: `d70098af827694a42e7bb31958cf9959ec3aceef0d432941960c15d0cb5091c8`

## Results

| mode | outcome | chunks | peak RSS delta | elapsed |
| --- | --- | ---: | ---: | ---: |
| full, unconstrained | success | 4096/4096 | 121.18 MiB | 4.76 s |
| full, +64 MiB headroom | **allocation failure** | 2181/4096 | 66.64 MiB | 2.43 s |
| streaming, +64 MiB headroom | **success** | 4096/4096 | 0.43 MiB | 8.97 s |

The unconstrained control and constrained streaming both produced the exact expected checksum `234473.0256500244`.

The constrained full-resident run failed after 2,181 chunks with:

`DefaultCPUAllocator: can't allocate memory ... Error code 12 (Cannot allocate memory)`

The constrained streaming run used the same fixed 64 MiB additional address-space headroom and completed all 4,096 chunks.

## Interpretation

S3 establishes the specific systems feasibility boundary the experiment was designed to test:

> Under this explicit Linux `RLIMIT_AS` configuration, the learned Canaria payload amplification workload cannot retain all logical chunks simultaneously, while one-chunk streaming completes correctly under the same address-space headroom.

This is stronger than an RSS-reduction observation because one mode becomes **infeasible** under the locked limit while the streaming mode remains feasible.

## Boundaries

The 4,096 chunks alternate the two already learned G7 compiler block states only to amplify memory measurement. They are **not** a trained 4,096-block model. S3 does not demonstrate a particular MCU, smartphone, SBC, or accelerator deployment, and it does not establish that arbitrary neural networks admit the same compact representation. It remains a systems/runtime-format result and must not be used as scientific compositional-generalization evidence.

Machine-readable result: `RESULT.json`.
