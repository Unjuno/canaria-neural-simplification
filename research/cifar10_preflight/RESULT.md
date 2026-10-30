# CIFAR-10 ResNet20 data/runtime preflight result

Completed 2026-10-26T09:23:38Z, exit 0. No formal teacher training, validation
inference, test inference, quality estimate, replacement experiment, CI, push,
charge, or local-PC use. Pre-execution protocol/code commit: 9154339.

- Full source-lookup-to-completion elapsed: 2100.343735 s
  (35m00s), under 45-minute expiry 09:33:38Z.
- Download-start to checksum-verified: 1928.789231 s (includes hash).
- Official Toronto archive: 170,498,071 bytes; MD5
  c58f30108f718f92721af3b95e74349a; SHA256
  6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce.
- Only five training members extracted in memory/deserialized. The test member
  was never opened/deserialized; archive transfer/hash/tar traversal necessarily
  processes compressed bytes containing it. No dataset files extracted to disk.
- Stratified splits: 40,000 teachertrain / 5,000 calibration / 5,000 validation;
  PCG64 seed20261026; exact indices and hashes retained. Normalization computed
  solely on teachertrain. Calibration/validation not evaluated.
- Model269,722 trainable parameters; float32 CPU4 intraop/1 interop;
  Python3.13.5, torch2.10.0+cpu, NumPy2.3.5. Seed260026 is preflight-only.
- 30 warmup +200 timed full batch128 updates; all shape/logit/loss/gradient
  finite checks passed. 200 updates: 40.722424383 s;
  4.911298947 updates/s; 628.646265 images/s.
- Peak RSS 831.023 MiB; final MemAvailable
  8,521,228 KiB. No GPU used.

## Budget projection, not a teacher-quality result

313 full-batch-equivalent updates/epoch ×120 epochs ×3 seeds =112,680 updates.
Measured-cost projection: 6.373059 h total,
2.124353 h per seed. This is not measured
whole-epoch runtime; final64 handling must be explicit in formal trainer.
Validation/checkpoint/epoch-boundary overhead is excluded; finite checks and
augmentation are included in measured update time. Short-run CPU throughput
may not reflect sustained runtime. No assertion that any accuracy gate passes.

A fresh72-hour ceiling measured from this preflight's first data action minus
measured preflight and projected teachers leaves
65.043512 h for
comparisons/validation/reporting. This is only an arithmetic ceiling; it does
not reset a pre-existing research deadline, allocate comparisons, or guarantee
capacity. Parent must apply the actual research start/deadline and overhead.

## Limits and next authority boundary

No remaining data/runtime blocker observed. Strict watchdog caveat: SIGALRM
was present, not an external process watchdog; the run completed nearly10min
before expiry so deadline enforcement was not exercised. Training determinism
across platforms was not tested (split reproducibility is explicit). No
checkpoint was saved or reused as a formal teacher. No further scientific run
is authorized by this preflight. Evidence contains raw200 timing samples,
full progress log, dependencies, source hashes, split indices, and data hashes.
