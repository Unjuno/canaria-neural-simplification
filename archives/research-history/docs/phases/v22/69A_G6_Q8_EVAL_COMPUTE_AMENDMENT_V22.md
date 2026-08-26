# G6 q8 evaluation compute amendment — v22

This amendment is operational only and does not change a scientific condition, model, codec, metric, seed, threshold, or trained parameter.

The original q8 follow-up evaluated greedy generation in four batches of 64 sequences. To reduce runtime after one completed q8 seed, the remaining seeds use one batch of 256 sequences for the same 256 test prompts.

Before use, the two implementations were compared on the same model/data and produced exactly identical generation-token accuracy, exact-sequence rate, and first-error-position mean.

The fast q8 runner also reuses the already locked FP32 confirmatory metrics from each seed JSON rather than recomputing them. Model training, compiler fitting, tau=8 repair, q8 packing, dequantization, and q8 evaluation are unchanged.