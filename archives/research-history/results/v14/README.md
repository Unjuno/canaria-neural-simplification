# v14 results index

Phase H performed a finer **bit-width × retained-weight-count** sweep and a short-repair follow-up.

Protocol: [`../../docs/phases/v14/33_PHASEH_PRECISION_COUNT_PROTOCOL_V14.md`](../../docs/phases/v14/33_PHASEH_PRECISION_COUNT_PROTOCOL_V14.md)

Key carried-forward observations:
- dense 5–12-bit Conv3 replacements were near the FP32 reference in the tested cohort;
- 4-bit retained most reference performance, while 2–3-bit exposed a sharper precision boundary;
- at fixed bit width, retained-weight count became the dominant bottleneck well before FP32 precision did;
- short repair reduced the K required for useful performance, but did not make very small K intrinsically sufficient.

See v15 for quantizer/refit controls that separate poor coefficient selection from true K insufficiency.