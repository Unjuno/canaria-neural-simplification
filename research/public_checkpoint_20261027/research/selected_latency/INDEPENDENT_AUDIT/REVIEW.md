> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Independent selected whole-model latency saved-evidence audit

PASS: 345 checks. Runtime d653a73406525b42c19c7266046f3cfa0317e56c; evidence commit 8229c4eb5fe803cde8ab01a8b3e53064a6b62368.

All 96 cells complete, 2,880 blocks, 20 iterations each, 30 rounds, all 16 seeds and both batches. Every row matches frozen alternating batch order and six permutations repeated five times. No exclusions. All 32 per-seed median/ratio rows and six geometric summaries reproduce exactly. Timer units and no timed loading/serialization checked in hash-bound source.

C/J geometric ratios: batch 1 = 0.8367120736129804; batch 128 = 0.9433388812895254. C lower in all 16 pairs in each batch. C/teacher = 0.698418340898702 and 0.5875424199697151; J/teacher = 0.8347176560783729 and 0.622832824580024. These are conditional observations, not hardware-general claims.



Environment snapshots agree on pinned versions and mkldnn false. Source asserts AVX2 and one intra/inter-op thread, sets seven thread/MKL environment controls plus ATEN AVX2, deterministic algorithms, eval/no_grad. Live thread counters were not separately serialized; this distinction is retained. CPU/process/load snapshots do not establish host isolation.

Variation: inclusive-quartile Tukey rule flags 177 of 2,880 blocks descriptively. Largest block/own-cell median is 25.235367447 (seed 3, batch 1, arm C; max 1793567.15 ns/forward, median 71073.55). Largest cell IQR/median is 0.275523046. All blocks retained; these post hoc descriptors do not invalidate the run or authorize retries. Shared-host contention and sequential-seed drift remain limitations.

No benchmark rerun, model inference, new dataset access, UCI/network, author-file edits, publication or user message. Audit only wrote its own directory. No training-time, new-data accuracy, runtime RAM or general CPU performance claim.
