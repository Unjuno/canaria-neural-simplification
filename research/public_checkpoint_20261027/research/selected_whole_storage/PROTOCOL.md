> 歴史的な事前計画の科学部分を抜粋した資料。本文の準備・未実行という時制は当時のもの。現在の実行状態は結果報告を参照。

> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。

# Frozen selected-cohort whole-model storage protocol

Hypothesis: previously selected replacement parameter ratio 0.788324 C/J overstates whole-model savings because identical stem, last two blocks and head remain. Evaluate all 16 fixed manifest IDs, no selection/retry/codec search. Descriptive ratios only, no success threshold. Missing/corrupt cases stay in failure census; incomplete denominator reported.

Read only existing 48 NPZ with allow_pickle=False; verify source file SHA256 and arm logical-state hash. No data, quarantine, inference, training, network or external writes. Whole teacher includes every source tensor. Whole C/J excludes every blocks.0.* and blocks.1.* teacher tensor, preserves stem.*, blocks.2.*, blocks.3.*, head.* names and exact numeric values; adds replacement.<original arm key>. No optimizer or unused replaced weights. Expected teacher: stem Linear(64,64), four blocks each LayerNorm(64), Linear(64,128), Linear(128,64), head Linear(64,10); C: w1/w2 biasless, J: two such maps, fixed selected width. All source dtype float32, preserved little-endian C-contiguous. Reject unexpected key/shape/dtype.

Serialization predetermined: sorted tensor names + '.npy', numpy NPY v1.0 allow_pickle=False; deterministic ZIP timestamp 1980-01-01, Unix create_system=3, external_attr=0600<<16. Both ZIP_STORED and ZIP_DEFLATED level6 once, same payload/schema, no bundled metadata. Preserve all 96 archives (16*3*2). Exact array readback, counts, dtype, shapes, source-to-destination mapping and hashes required. Byte count actual filesystem size, separately sum tensor payload bytes and parameters (all tensors here are parameters). Record all per-seed teacher/C/J counts, archive bytes, C/J,C/teacher,J/teacher ratios for each codec and parameter counts; aggregate geometric means,min,max,smaller/tie/larger. No latency/RAM/runtime-deployment conclusions.

Bounds: maximum 600 elapsed seconds, 256MiB address space, one local process, numpy/stdlib only, OPENBLAS_NUM_THREADS=1; input/artifact ceiling 128MiB, each seed failure logged without dropping cohort. Independent lightweight checker after results, not new fit/experiment. Preserve protocol/runtime commits, environment, source hashes, raw rows, failure census and report. No GitHub writes. This is an analysis serialization, NOT validated runnable deployment.
