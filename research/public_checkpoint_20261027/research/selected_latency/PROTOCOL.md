> 歴史的な事前計画の科学部分を抜粋した資料。本文の準備・未実行という時制は当時のもの。現在の実行状態は結果報告を参照。

> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Frozen selected whole-model latency protocol
Source storage acceptance 90629b18fe007b95dc089321022c4d1f73e3f2a9; exact 16 teacher and 32 selected C/J stored NPZ whole-model archives and manifests. No new selection, training, external data, original 450-test metrics, CI, publication, paid services or local PC.
All 48 deployed graphs must exactly match archived tensor keys, counts and source hashes, and agree within absolute 1e-5 with saved 270-validation logits, in original chunks 256 and14, before any timing. Labels are not consumed.
Locked Python3.13.5 torch2.10.0+cpu numpy2.3.5 sklearn1.8.0 scipy1.17.0; CPU AVX2, MKL COMPATIBLE, mkldnn disabled, deterministic algorithms, one intra/inter-op thread, eval and no_grad. No compilation, quantization, kernel tuning or affinity tuning.
Timing uses fixed synthetic normalized inputs: numpy default_rng(20261027).integers(0,17,(128,64)).astype(float32)/16; batch1 is its first row. Same input across all models and seeds. Input generation, loading, hashing, correctness, serialization and summary excluded from timed blocks. perf_counter_ns; 20 warmup forwards per model/batch. 30 rounds, 20 forwards per timed block. Each seed/batch cycles through all six lexicographic permutations of teacher/C/J five times, offset by seed modulo6. Seeds ascending; batch order alternates by seed. Raw block total ns and ns/forward for every arm/seed/batch/round/order. No adaptive iterations or repeat run.
Per-seed latency is median of 30 block ns/forward values. Report all16 paired C/J, C/teacher, J/teacher ratios for both batches and their geometric means, minima/maxima and sign counts; no selective exclusions, no unsupported hardware-general speed claim or training-time claim. No significance test. Shared environment may be contended: capture CPU, affinity, load averages and process census before/after, explicitly caveat interference and sequential-seed temporal drift.

