# CIFAR-10 ResNet-20 runtime/data preflight (not a quality experiment)
Frozen before execution, 2026-10-26. Base c7fd4fb. No CI/push/charges/local-PC.
Conservative first data action: official source lookup 2026-10-26T08:48:38Z.
Hard expiry: 2026-10-26T09:33:38Z (45 minutes including download).
Official primary source https://cave.cs.toronto.edu/kriz/cifar.html confirms Python
archive MD5 c58f30108f718f92721af3b95e74349a and five 10k train batches.
Download https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz only.
Check MD5 before processing; also record SHA256. Never extract archive to disk.
Validate tar paths/types/sizes; extractfile only data_batch_1..5. Never access
or deserialize test_batch (compressed archive transfer/hash necessarily covers its bytes).
Restricted pickle globals: numpy ndarray/dtype/_reconstruct only; reject others.
Check uint8 [10000,3072], 10k integer labels0..9, combined 5000/class.
PCG64 split seed20261026: in class order0..9, permute class indices, take
4000 teachertrain,500 calibration,500 validation. Store int64 indices and hashes.
Normalization mean/std computed ONLY teachertrain, streaming chunks.
Calibration/validation images never used in training/evaluation. No test accuracy.
ResNet20: 3x3 stem16, three blocks/stage16/32/64, each two3x3 conv+BN,
option-A stride2 and symmetric zero-channel-padding shortcuts, global average,
10-class linear. He initialization; BN weight1/bias0. Conventional 6n+2 depth.
Preflight seed260026 (not a fresh main cohort); float32 CPU, FOUR intraop threads,
one interop thread fixed before timing. Batch128 shuffled train without replacement;
pad4 zero/random crop32/horizontal flip p.5; normalize by teachertrain stats.
SGD lr.1,momentum.9,weight_decay5e-4. Exactly30 warmup+200 timed updates,
with finite logits/loss/grad checks. Timing includes augmentation and checks.
No accuracy/quality threshold estimates. Report total params, dependency versions,
memory, elapsed time, per-step latencies, throughput, and estimated 120epoch
teacher time with ceil(40000/128)=313 updates/epoch times3 seeds. Projection only,
excludes validation/checkpoint overhead; no formal teacher or replacement run.
Explicit remaining 72h ceiling after measured preflight + teacher projection,
not a guaranteed experiment budget. Report deadline-derived budget if provided.
All failures/expiry are blocked/partial, not success. No checksum relaxation.
