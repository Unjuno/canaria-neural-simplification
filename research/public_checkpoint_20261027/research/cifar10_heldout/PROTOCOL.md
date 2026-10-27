> 歴史的な事前計画の科学部分を抜粋した資料。本文の準備・未実行という時制は当時のもの。現在の実行状態は結果報告を参照。

> 公開用抜粋。元の完全な実行資料とは区別する。変換と原 SHA256 はルート PROVENANCE.json を参照。


Source/synthetic preparation only. No real data/checkpoints were inspected. No
user-announced plan and a current reserved four-thread CPU slot (global eight).
one global run record, then starts the shared original 2h / Oct29 06:50 UTC timer
before numerical imports, hashing artifacts or data loading. The replacement
worker lock prevents concurrent replacement/evaluation. No resume/retry: a failed
attempt consumes the record and requires a separately reviewed decision; never
delete it to obtain another look. SIGALRM may terminate without an exit receipt;

All seeds are admitted before test extraction. Pinned preflight training data are
used solely to reconstruct already-frozen normalization/split identity and reuse
teacher admission; no fitting. The test archive is MD5/SHA256 checked on the same
open descriptor that is rewound for safe tar traversal and restricted unpickling.
Every model sees identical 10,000 images in original order, without augmentation,
adaptation or updates; all modules EVAL and inference_mode; finite checks and
state hashes guard adaptation. Predictions, labels and original image IDs are
retained with hashes and source/data/checkpoint provenance. Summaries give exact
correct counts, paired disagreement counts, degradation pp, descriptive <=2pp
(teacher correct minus 200), and all three pairwise arm contrasts. Endpoint NMSE
uses summed squared errors divided by full-N per-feature centered teacher SS plus
1e-12 times N*C*H*W, not average batch ratios. No statistical noninferiority or
generalization claim; no threshold or recipe changes after a heldout look.

Frozen-fit admission independently reconstructs the 1000-batch schedule from
and the exact Adam param-group recipe. Each saved parameter must have finite,
correctly shaped moments and step1000; cursor1000 alone is insufficient. These
checks establish artifact consistency, not proof of training history. One-look
of requested output directory.
