# PROVISIONAL heldout evaluator — not released, not run

Source/synthetic preparation only. No real data/checkpoints were inspected. No
release manifest is supplied here. Actual use needs separate parent-reviewed,
user-announced plan and a current reserved four-thread CPU slot (global eight).
The standalone stdlib entry verifies source/release/slot first, permanently claims
one global run record, then starts the shared original 2h / Oct29 06:50 UTC timer
before numerical imports, hashing artifacts or data loading. The replacement
worker lock prevents concurrent replacement/evaluation. No resume/retry: a failed
attempt consumes the record and requires a separately reviewed decision; never
delete it to obtain another look. SIGALRM may terminate without an exit receipt;
the durable launch and one-run registry remain evidence of an attempted look.

Release must bind all three planned outcomes and each qualified seed's fixed-final
teacher epoch120, student update1000, recipe, validation, export, metrics hashes,
normalization/split identity, complete evaluated source and fit-source pins.
Default is blocked on any unresolved failure. Explicit exclusions require a
parent-predeclared failure policy artifact, hash-bound failure evidence and resolved
classification; every failed seed remains in the report. The parent must verify
that the failure declaration truly preceded outcomes (hashes are not authentication).
Never substitute seeds or claim the qualified subset represents all planned seeds.

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

Parent still must review code, freeze artifacts, attest predeclaration provenance,
reserve CPU, and issue release manifest. Preparation does not authorize any run.

Frozen-fit admission independently reconstructs the 1000-batch schedule from
approved calibration IDs and seed, initial tensors via the fixed arm constructors,
and the exact Adam param-group recipe. Each saved parameter must have finite,
correctly shaped moments and step1000; cursor1000 alone is insufficient. These
checks establish artifact consistency, not proof of training history. One-look
registry creation fsyncs both the exclusive file and its parent directory before
bootstrap/test access; any existing or partial registry fails closed regardless
of requested output directory.
