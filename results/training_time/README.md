# Training-time consolidation results index

This directory indexes the corrected Canaria mainline experiments G7–G17.

## Files

- `summary.json` — compact machine-readable headline outcomes.
- `protocol_manifest.json` — fresh seed ranges, decision rules, and available SHA256 values for preregistered protocol locks.

## Confirmatory sequence

| experiment | question | fresh seeds | decision |
|---|---|---|---|
| G7 | progressive consolidation vs small-from-start / one-shot | 4300–4307 | PASS |
| G8 | does correct function-aligned transfer matter? | 4500–4507 | PASS |
| G9 | how much transfer fit is useful? | 4700–4707 | PASS |
| G10 | can structured weight inheritance replace functional fitting? | 4900–4907 | PASS: inheritance alone insufficient; hybrid best |
| G11 | can a calibration-only controller autonomously reach the target architecture? | 5400–5407 | PASS |
| G15 | staged 4→3→2 vs waiting for direct 4→2 | 5800–5807 | PASS |
| G17 | does fit factorization alone reproduce staged benefit? | 6000–6007 | PASS equivalence: no |

## Interpretation boundary

The strongest current mechanism separation is G15 + G17:

- staged consolidation with **task learning between commits** beats direct consolidation;
- the same 4→3→2 compiler factorization **without task learning between fits** is equivalent to direct 4→2.

This supports a recontracting / training-path adaptation interpretation in the tested small real-text LM. It does not establish the internal cause of recontracting or generalization to large pretrained models.

Raw run records and exact lock files remain in the corresponding research handoff archives. This repository index intentionally does not manufacture missing hashes or reclassify exploratory runs as confirmatory.
